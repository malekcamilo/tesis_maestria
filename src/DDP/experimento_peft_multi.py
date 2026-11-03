import os

import argparse
import contextlib
import faulthandler
import json
import logging
import re
import socket
import time
import torch
import unicodedata
import torch.distributed as dist

from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Configuración MLOps 2026: Logging Estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Práctica 2026: Clases inmutables para el dominio
class Categoria(str, Enum):
    CONFERENCIA = "objeto de conferencia"
    APRENDIZAJE = "objeto de aprendizaje"
    ARTICULO = "articulo"
    TESIS = "tesis"
    LIBRO = "libro"
    OTRO = "otro"
    SIN_CLASIFICAR = "sin_clasificar"

VALID_CATEGORIES_LIST = [c.value for c in Categoria if c != Categoria.SIN_CLASIFICAR]

def normalizar_texto(texto: str) -> str:
    texto = str(texto).lower().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def extraer_categoria(texto: str) -> str:
    texto_norm = normalizar_texto(texto)
    for category in VALID_CATEGORIES_LIST:
        if category in texto_norm:
            return category
    return Categoria.SIN_CLASIFICAR.value

def format_prompt(example: Dict[str, Any], is_training: bool = False, eos_token: str = "<|eot_id|>") -> str:
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Eres un clasificador automático de documentos académicos. "
        "Tu única tarea es asignar la categoría correcta a los registros bibliográficos provistos.\n"
        "Regla estricta: Debes responder ÚNICAMENTE con el nombre exacto de la categoría. No incluyas explicaciones, puntuación adicional ni texto conversacional.\n"
        "Categorías válidas: Articulo, Objeto de conferencia, Tesis, Libro, Otro, Objeto de aprendizaje.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Clasifica el siguiente registro:\n"
        f"<titulo>{example['title']}</titulo>\n"
        f"<resumen>{example['abstract']}</resumen><|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if is_training:
        prompt += example['type'] + eos_token
    return prompt

def setup_distributed_xpu() -> int:
    # Rank desde Intel MPI (mpirun) o Slurm (srun); el process group lo crea accelerate.
    if "PMI_RANK" in os.environ:
        rank, world_size, local_rank = os.environ["PMI_RANK"], os.environ["PMI_SIZE"], os.environ["MPI_LOCALRANKID"]
    elif "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        rank, world_size, local_rank = os.environ["SLURM_PROCID"], os.environ["SLURM_NTASKS"], os.environ["SLURM_LOCALID"]
    else:
        logger.warning("No se detectó entorno MPI. Ejecutando en modo XPU único.")
        torch.xpu.set_device(0)
        return 0

    os.environ["RANK"] = rank
    os.environ["WORLD_SIZE"] = world_size
    os.environ["LOCAL_RANK"] = local_rank
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    xpu_index = int(local_rank)
    torch.xpu.set_device(xpu_index)
    # Inicializa el contexto en el tile propio
    torch.zeros(1, device=f"xpu:{xpu_index}")
    cpus = sorted(os.sched_getaffinity(0))
    logger.info(f"Rank {rank}/{world_size} | host={socket.gethostname()} | ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')} "
                f"| xpu:{xpu_index} de {torch.xpu.device_count()} | CPUs {cpus[0]}-{cpus[-1]} ({len(cpus)})")
    return xpu_index

def evaluate_model(model: Any, tokenizer: PreTrainedTokenizer, test_dataset: Any) -> Tuple[float, float, float, Dict[str, float], Dict[str, Any], float, float, float]:
    model.eval()
    predictions: List[str] = []
    references: List[str] = []
    
    exact_matches = 0
    exact_matches_per_class = {cat: 0 for cat in VALID_CATEGORIES_LIST}
    total_per_class = {cat: 0 for cat in VALID_CATEGORIES_LIST}

    tokenizer.padding_side = "left"
    batch_size = 16 
    
    dataset_list = list(test_dataset)
    
    logger.info(f"Iniciando inferencia vectorizada sobre test (Batch Size: {batch_size})...")
    
    start_time = time.time()
    total_generated_tokens = 0
    
    for i in range(0, len(dataset_list), batch_size):
        batch = dataset_list[i:i+batch_size]
        prompts = [format_prompt(item, is_training=False) for item in batch]
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True, 
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True
            )

        for j, item in enumerate(batch):
            input_length = inputs.input_ids[j].shape[0]
            generated_length = outputs[j].shape[0] - input_length
            total_generated_tokens += generated_length
            
            generated_text = tokenizer.decode(outputs[j][input_length:], skip_special_tokens=True).strip()
            raw_pred = generated_text.split('\n')[0].strip()

            norm_true = normalizar_texto(item['type'])
            
            if norm_true in total_per_class:
                total_per_class[norm_true] += 1
            
            if normalizar_texto(raw_pred) == norm_true:
                exact_matches += 1
                if norm_true in exact_matches_per_class:
                    exact_matches_per_class[norm_true] += 1

            predictions.append(extraer_categoria(generated_text))
            references.append(norm_true)

    tokenizer.padding_side = "right"

    inference_time = time.time() - start_time
    total = len(dataset_list)
    samples_per_second = total / inference_time if inference_time > 0 else 0
    tokens_per_second = total_generated_tokens / inference_time if inference_time > 0 else 0

    em_global = exact_matches / total if total > 0 else 0.0
    acc = accuracy_score(references, predictions)
    macro_f1 = f1_score(references, predictions, average='macro', zero_division=0)
    
    em_per_class: Dict[str, float] = {}
    for cat in VALID_CATEGORIES_LIST:
        if total_per_class[cat] > 0:
            em_per_class[cat] = exact_matches_per_class[cat] / total_per_class[cat]
        else:
            em_per_class[cat] = 0.0
            
    report = classification_report(references, predictions, output_dict=True, zero_division=0)
    
    return acc, macro_f1, em_global, em_per_class, report, inference_time, samples_per_second, tokens_per_second

class StepTimerCallback(TrainerCallback):
    """Tiempo de cada paso (forward, backward, allreduce y optimizer) en todos los ranks."""
    def __init__(self, xpu_index: int):
        self.xpu_index = xpu_index
        self.times: List[float] = []
        self._t0 = 0.0
        # Perro guardián: traceback y salida si un paso supera STEP_TIMEOUT_S (0 = desactivado)
        self.timeout_s = int(os.environ.get("STEP_TIMEOUT_S", "600"))

    def on_step_begin(self, args, state, control, **kwargs):
        if self.timeout_s > 0:
            faulthandler.dump_traceback_later(self.timeout_s, exit=True)
        torch.xpu.synchronize(self.xpu_index)
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.xpu.synchronize(self.xpu_index)
        self.times.append(time.perf_counter() - self._t0)
        if self.timeout_s > 0:
            faulthandler.cancel_dump_traceback_later()

    def resumen(self, descartar: int = 10) -> Dict[str, float]:
        # Los primeros pasos incluyen compilación JIT y warmup de oneCCL
        t = sorted(self.times[descartar:] if len(self.times) > descartar else self.times)
        if not t:
            return {}
        return {
            "steps_medidos": len(t),
            "mediana_s": round(t[len(t) // 2], 3),
            "media_s": round(sum(t) / len(t), 3),
            "p90_s": round(t[int(0.9 * (len(t) - 1))], 3),
        }

def medir_comunicacion(model: Any, xpu_index: int, iters: int = 10, bucket_mb: int = 25) -> Dict[str, Any]:
    """Allreduce aislado de los gradientes en buckets de 25 MB: cota superior de la comunicación por paso."""
    params = [p for p in model.parameters() if p.requires_grad]
    dtype = params[0].dtype
    n_elem = sum(p.numel() for p in params)
    elem_size = torch.tensor([], dtype=dtype).element_size()
    por_bucket = bucket_mb * 2**20 // elem_size
    buckets = [torch.ones(min(por_bucket, n_elem - i), dtype=dtype, device=f"xpu:{xpu_index}")
               for i in range(0, n_elem, por_bucket)]
    for b in buckets:
        dist.all_reduce(b)
    torch.xpu.synchronize(xpu_index)
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        for b in buckets:
            dist.all_reduce(b)
    torch.xpu.synchronize(xpu_index)
    dt = torch.tensor([(time.perf_counter() - t0) / iters], device=f"xpu:{xpu_index}")
    dist.all_reduce(dt, op=dist.ReduceOp.MAX)
    return {"gradientes_mb": round(n_elem * elem_size / 2**20, 1), "dtype": str(dtype).replace("torch.", ""),
            "buckets": len(buckets), "allreduce_por_paso_s": round(dt.item(), 3)}

def medir_comunicacion_expuesta(trainer: Any, xpu_index: int, pasos: int = 8, warmup: int = 2) -> Dict[str, Any]:
    """Comunicación expuesta: forward+backward con y sin no_sync(). No modifica el modelo."""
    ddp = trainer.model_wrapped
    if not isinstance(ddp, torch.nn.parallel.DistributedDataParallel):
        return {}
    dispositivo = f"xpu:{xpu_index}"
    batch = next(iter(trainer.get_train_dataloader()))
    batch = {k: v.to(dispositivo) if torch.is_tensor(v) else v for k, v in batch.items()}
    ddp.train()

    def paso(sincronizar: bool) -> float:
        ddp.zero_grad(set_to_none=True)
        torch.xpu.synchronize(xpu_index)
        t0 = time.perf_counter()
        # no_sync() tiene que envolver el forward y el backward
        with (contextlib.nullcontext() if sincronizar else ddp.no_sync()):
            with trainer.compute_loss_context_manager():
                loss = trainer.compute_loss(ddp, batch)
            loss.backward()
        torch.xpu.synchronize(xpu_index)
        return time.perf_counter() - t0

    con_sync, sin_sync = [], []
    for _ in range(warmup + pasos):  # alternados, para que ambos vean las mismas condiciones
        con_sync.append(paso(True))
        sin_sync.append(paso(False))
    ddp.zero_grad(set_to_none=True)

    mediana = lambda xs: sorted(xs)[len(xs) // 2]
    t = torch.tensor([mediana(con_sync[warmup:]), mediana(sin_sync[warmup:])], device=dispositivo)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    con_s, sin_s = t.tolist()
    return {"pasos_medidos": pasos, "fwd_bwd_con_sync_s": round(con_s, 3), "fwd_bwd_sin_sync_s": round(sin_s, 3),
            "expuesta_s": round(con_s - sin_s, 3)}

def ejecutar_entrenamiento(args: argparse.Namespace, tokenizer: PreTrainedTokenizer, suffix: str, xpu_index: int, n_nodes: int, g_gpus: int) -> None:
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info("Procesando configuración inicial (Entrenamiento)...")

    datasets = load_dataset("json", data_files={
        "train": "train_tesis.jsonl",
        "val": "val_tesis.jsonl"
    })

    def formatting_prompts_func(example: Dict[str, Any]) -> str:
        return format_prompt(example, is_training=True, eos_token=tokenizer.eos_token)

    if args.method == "base":
        if int(os.environ.get("RANK", 0)) == 0:
            logger.warning("El método 'base' no requiere entrenamiento.")
        return

    device_map = {"": f"xpu:{xpu_index}"}
    device_name = torch.xpu.get_device_name(xpu_index)
    prop = torch.xpu.get_device_properties(xpu_index)
    vram_gb = prop.total_memory / (1024**3)

    logger.info(f"[TELEMETRÍA XPU] Dispositivo: xpu:{xpu_index} -> {device_name} | VRAM: {vram_gb:.1f} GB")

    bnb_config = None
    if args.method in ["qlora", "qdora"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=bnb_config
    )

    tasa_aprendizaje = 2e-5 
    target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj"] if args.target_modules == "attn" else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if args.method in ["lora", "dora", "qlora", "qdora"]:
        model.enable_input_require_grads()
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.1,
            target_modules=target_modules_list,
            use_dora=(args.method in ["dora", "qdora"])
        )
        model = get_peft_model(model, peft_config)
        tasa_aprendizaje = 2e-4

    per_device_train_bs = 8
    per_device_eval_bs = 8
    grad_accum = 1

    training_args = SFTConfig(
        output_dir=f"./resultados_{suffix}",
        learning_rate=tasa_aprendizaje,
        seed=args.seed,
        per_device_train_batch_size=per_device_train_bs,
        per_device_eval_batch_size=per_device_eval_bs,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=5,
        max_steps=args.max_steps,
        bf16=True,
        save_strategy="no",
        logging_strategy="epoch",
        eval_strategy="no" if args.max_steps > 0 else "epoch",
        max_length=512,
        ddp_backend="xccl" if "RANK" in os.environ else None,
        ddp_find_unused_parameters=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        args=training_args,
        formatting_func=formatting_prompts_func
    )

    step_timer = StepTimerCallback(xpu_index)
    trainer.add_callback(step_timer)

    try:
        torch.xpu.reset_peak_memory_stats(xpu_index)
    except (AttributeError, TypeError):
        pass
    
    start_time = time.time()
    try:
        mem_before_gb = torch.xpu.memory_allocated() / (1024**3)
        logger.info(f"[TELEMETRÍA XPU] Memoria base ocupada por el modelo: {mem_before_gb:.2f} GB")
    except AttributeError:
        mem_before_gb = 0.0
        
    train_result = trainer.train()

    step_time_por_rank = [step_timer.resumen()]
    if dist.is_initialized():
        step_time_por_rank = [None] * dist.get_world_size()
        dist.all_gather_object(step_time_por_rank, step_timer.resumen())

    train_time = (time.time() - start_time) / 3600
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info("Entrenamiento completado.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    param_percent = 100 * trainable_params / total_params
    
    samples_per_sec = train_result.metrics.get("train_samples_per_second", 0.0)
    steps_per_sec = train_result.metrics.get("train_steps_per_second", 0.0)

    try:
        memory_gb = torch.xpu.max_memory_allocated() / (1024**3)
        reserved_gb = torch.xpu.max_memory_reserved() / (1024**3)
    except AttributeError:
        memory_gb = 0.0
        reserved_gb = 0.0

    if int(os.environ.get("RANK", 0)) == 0:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        log_path = logs_dir / f"loss_{suffix}.json"
        with open(log_path, 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2)

        if args.method in ["lora", "dora", "qlora", "qdora"]:
            trainer.model.save_pretrained(f"./resultados_{suffix}")
        else:
            trainer.save_model(f"./resultados_{suffix}")
        
        result_path = logs_dir / f"metricas_train_{suffix}.json"
        
        final_train_loss = next((log.get("loss") for log in reversed(trainer.state.log_history) if "loss" in log), "N/A")
        final_eval_loss = next((log.get("eval_loss") for log in reversed(trainer.state.log_history) if "eval_loss" in log), "N/A")
        
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        global_batch_size = per_device_train_bs * grad_accum * world_size
        
        resultados_train = {
            "config": {
                "fase": "train",
                "method": args.method,
                "lora_r": args.lora_r if args.method in ["lora", "dora", "qlora", "qdora"] else None,
                "target_modules": args.target_modules,
                "seed": args.seed,
                "n_nodes": n_nodes,
                "g_gpus_per_node": g_gpus,
                "world_size": world_size,
                "global_batch_size": global_batch_size
            },
            "metrics": {
                "train_time_hours": round(train_time, 2),
                "base_memory_gb": round(mem_before_gb, 2),
                "peak_memory_gb": round(memory_gb, 2),
                "peak_reserved_memory_gb": round(reserved_gb, 2),
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "trainable_percent": round(param_percent, 4),
                "train_samples_per_second": round(samples_per_sec, 2),
                "train_steps_per_second": round(steps_per_sec, 2),
                "final_train_loss": round(final_train_loss, 4) if isinstance(final_train_loss, float) else final_train_loss,
                "final_eval_loss": round(final_eval_loss, 4) if isinstance(final_eval_loss, float) else final_eval_loss,
                "step_time": step_time_por_rank[0],
                "step_time_mediana_max_entre_ranks_s": max((r.get("mediana_s", 0.0) for r in step_time_por_rank), default=0.0),
                "step_time_por_rank": step_time_por_rank
            }
        }
        with open(result_path, 'w') as f:
            json.dump(resultados_train, f, indent=2)

        t_loss_str = f"{final_train_loss:.4f}" if isinstance(final_train_loss, float) else str(final_train_loss)
        e_loss_str = f"{final_eval_loss:.4f}" if isinstance(final_eval_loss, float) else str(final_eval_loss)
        
        logger.info("\n" + "=" * 50)
        logger.info(f"{'REPORTE DE ENTRENAMIENTO':^50}")
        logger.info("=" * 50)
        logger.info(f"{'Método:':<12} {args.method.upper()}")
        logger.info(f"{'Tiempo:':<12} {train_time:.2f} horas")
        logger.info(f"{'Memoria Asig:':<12} {memory_gb:.2f} GB")
        logger.info(f"{'Memoria Resv:':<12} {reserved_gb:.2f} GB")
        logger.info(f"{'Parám. Entr:':<12} {trainable_params:,} ({param_percent:.2f}%)")
        logger.info(f"{'Throughput:':<12} {samples_per_sec:.2f} samples/s")
        if step_time_por_rank[0]:
            st = step_time_por_rank[0]
            logger.info(f"{'Step time:':<12} mediana {st['mediana_s']:.2f}s | p90 {st['p90_s']:.2f}s ({st['steps_medidos']} steps, rank 0)")
        logger.info(f"{'Train Loss:':<12} {t_loss_str}")
        logger.info(f"{'Eval Loss:':<12} {e_loss_str}")
        logger.info(f"{'Pesos:':<12} ./resultados_{suffix}")
        logger.info(f"{'Historial:':<12} {log_path}")
        logger.info(f"{'Métricas:':<12} {result_path}")
        logger.info("=" * 50 + "\n")

    # Medición de comunicación, después de guardar los resultados (fuera de train_time)
    if dist.is_initialized() and dist.get_world_size() > 1:
        if step_timer.timeout_s > 0:
            faulthandler.dump_traceback_later(2 * step_timer.timeout_s, exit=True)
        try:
            comunicacion = medir_comunicacion(model, xpu_index)
            comunicacion.update(medir_comunicacion_expuesta(trainer, xpu_index))
        except Exception as e:
            logger.warning(f"No se pudo medir la comunicación: {e}")
            comunicacion = {}
        if step_timer.timeout_s > 0:
            faulthandler.cancel_dump_traceback_later()
        step_med = max((r.get("mediana_s", 0.0) for r in step_time_por_rank), default=0.0)
        if comunicacion and step_med > 0:
            comunicacion["allreduce_fraccion_del_paso"] = round(comunicacion["allreduce_por_paso_s"] / step_med, 4)
            if "expuesta_s" in comunicacion:
                comunicacion["expuesta_fraccion_del_paso"] = round(comunicacion["expuesta_s"] / step_med, 4)
        if comunicacion and int(os.environ.get("RANK", 0)) == 0:
            with open(result_path) as f:
                resultados_train = json.load(f)
            resultados_train["metrics"]["comunicacion"] = comunicacion
            with open(result_path, 'w') as f:
                json.dump(resultados_train, f, indent=2)
            c = comunicacion
            logger.info(f"{'Comunic.:':<12} allreduce aislado {c['allreduce_por_paso_s']:.3f}s/paso "
                        f"({c['gradientes_mb']} MB {c['dtype']}, {c['buckets']} buckets) = "
                        f"{100 * c.get('allreduce_fraccion_del_paso', 0):.1f}% del paso (cota superior, sin solapamiento)")
            if "expuesta_s" in c:
                logger.info(f"{'Comunic.:':<12} expuesta {c['expuesta_s']:.3f}s/paso = "
                            f"{100 * c.get('expuesta_fraccion_del_paso', 0):.1f}% del paso "
                            f"(fwd+bwd con sync {c['fwd_bwd_con_sync_s']:.3f}s vs sin sync {c['fwd_bwd_sin_sync_s']:.3f}s, "
                            f"{c['pasos_medidos']} pasos c/u)")

def ejecutar_evaluacion(args: argparse.Namespace, tokenizer: PreTrainedTokenizer, suffix: str, xpu_index: int, n_nodes: int, g_gpus: int) -> None:
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info("Procesando configuración inicial (Evaluación)...")

    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        if int(os.environ.get("RANK", 0)) != 0:
            # Los workers secundarios esperan al maestro pasivamente
            if dist.is_initialized():
                dist.barrier()
            return
        logger.warning("La inferencia standalone custom no fragmenta el dataset. Se ejecutará centralizadamente en RANK 0.")

    datasets = load_dataset("json", data_files={
        "test": "test_tesis.jsonl"
    })

    device_map = {"": f"xpu:{xpu_index}"}

    if args.method == "base":
        logger.info("Evaluando modelo base sin fine-tuning...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map=device_map
        )
    elif args.method in ["lora", "dora", "qlora", "qdora"]:
        logger.info(f"Cargando modelo base y acoplando adaptador {args.method.upper()} desde ./resultados_{suffix}")
        bnb_config = None
        if args.method in ["qlora", "qdora"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=bnb_config
        )
        model = PeftModel.from_pretrained(base_model, f"./resultados_{suffix}")
    else:
        logger.info(f"Cargando modelo completo ({args.method.upper()}) desde ./resultados_{suffix}")
        model = AutoModelForCausalLM.from_pretrained(
            f"./resultados_{suffix}",
            dtype=torch.bfloat16,
            device_map=device_map
        )

    acc, macro_f1, em_global, em_per_class, report, inference_time, samples_per_sec, tokens_per_sec = evaluate_model(model, tokenizer, datasets["test"])

    if int(os.environ.get("RANK", 0)) == 0:
        logger.info("\n" + "=" * 50)
        logger.info(f"{'REPORTE DE EVALUACIÓN':^50}")
        logger.info("=" * 50)
        logger.info(f"{'Método:':<12} {args.method.upper()}")
        logger.info(f"{'Accuracy:':<12} {acc:.4f}")
        logger.info(f"{'Macro F1:':<12} {macro_f1:.4f}")
        logger.info(f"{'Global EM:':<12} {em_global:.4f}")
        logger.info(f"{'Latencia:':<12} {inference_time:.2f}s total ({samples_per_sec:.2f} doc/s | {tokens_per_sec:.2f} tok/s)")
        
        logger.info("-" * 50)
        logger.info(f"{'MÉTRICAS POR CLASE':^50}")
        logger.info("-" * 50)
        logger.info(f"{'Clase':<25} | {'F1-Score':<10} | {'Exact Match':<10}")
        logger.info("-" * 50)
        for cat in VALID_CATEGORIES_LIST:
            f1_c = report.get(cat, {}).get('f1-score', 0.0)
            em_c = em_per_class.get(cat, 0.0)
            logger.info(f"{cat.title():<25} | {f1_c:.4f}     | {em_c:.4f}")
        
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        result_path = logs_dir / f"metricas_eval_{suffix}.json"
        logger.info("=" * 50)
        logger.info(f"{'Archivo:':<12} {result_path}")
        logger.info("=" * 50 + "\n")

        resultados_eval = {
            "config": {
                "fase": "eval",
                "method": args.method,
                "lora_r": args.lora_r if args.method in ["lora", "dora", "qlora", "qdora"] else None,
                "target_modules": args.target_modules,
                "seed": args.seed,
                "allocated_nodes": n_nodes,
                "allocated_gpus_per_node": g_gpus,
                "effective_evaluation_nodes": 1,
                "effective_evaluation_gpus": 1
            },
            "metrics": {
                "global": {
                    "accuracy": round(acc, 4),
                    "macro_f1": round(macro_f1, 4),
                    "exact_match": round(em_global, 4),
                    "inference_time_seconds": round(inference_time, 2),
                    "samples_per_second": round(samples_per_sec, 2),
                    "tokens_per_second": round(tokens_per_sec, 2)
                },
                "per_class": {}
            }
        }
        
        for cat in VALID_CATEGORIES_LIST:
            resultados_eval["metrics"]["per_class"][cat] = {
                "f1_score": round(report.get(cat, {}).get('f1-score', 0.0), 4),
                "exact_match": round(em_per_class.get(cat, 0.0), 4),
                "precision": round(report.get(cat, {}).get('precision', 0.0), 4),
                "recall": round(report.get(cat, {}).get('recall', 0.0), 4),
                "support": report.get(cat, {}).get('support', 0)
            }

        with open(result_path, 'w') as f:
            json.dump(resultados_eval, f, indent=2)

    if int(os.environ.get("WORLD_SIZE", 1)) > 1 and dist.is_initialized():
        dist.barrier()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--accion", type=str, choices=["train", "eval"], default="train", help="Fase a ejecutar de manera aislada (por defecto: train).")
    parser.add_argument("--method", type=str, choices=["base", "fft", "lora", "dora", "qlora", "qdora"], required=True)
    parser.add_argument("--lora_r", type=int, default=8, help="Rango de LoRA (alpha = 2*r)")
    parser.add_argument("--target_modules", type=str, default="all", choices=["all", "attn"], help="Módulos a intervenir: 'all' (completo) o 'attn' (solo atención)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Corta el entrenamiento tras N steps y desactiva la evaluación (medición de overhead DDP). -1 = entrenamiento completo.")
    parser.add_argument("--topologia", type=str, default=None, help="Topología del entrenamiento a usar en el nombre de resultados_* (ej. n2_g4). Por defecto, la del job actual. Útil para evaluar con 1 GPU un modelo entrenado en varios nodos.")
    args = parser.parse_args()
    if args.topologia is not None and not re.fullmatch(r"n\d+_g\d+", args.topologia):
        parser.error("--topologia debe tener la forma n<nodos>_g<gpus por nodo>, por ejemplo n2_g4")

    info_extra = ""
    if args.method in ["lora", "dora", "qlora", "qdora"]:
        info_extra = f" | Rank={args.lora_r}"

    if not torch.xpu.is_available():
        raise RuntimeError("XPU no disponible. Este script requiere Intel XPU.")
        
    xpu_index = setup_distributed_xpu()

    if int(os.environ.get("RANK", 0)) == 0:
        logger.info("\n" + "=" * 58)
        logger.info(f"Iniciando Trabajo: Acción={args.accion} | Método={args.method}{info_extra}")
        logger.info(f"Target Modules: {args.target_modules}")
        logger.info("=" * 58)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_nodes = int(os.environ.get("SLURM_NNODES", "1"))
    g_gpus = int(os.environ.get("SLURM_NTASKS_PER_NODE", "1"))
    topo_id = args.topologia or f"n{n_nodes}_g{g_gpus}"

    if args.method in ["lora", "dora", "qlora", "qdora"]:
        suffix = f"{args.method}_r{args.lora_r}_{args.target_modules}_s{args.seed}_{topo_id}"
    else:
        suffix = f"{args.method}_s{args.seed}_{topo_id}"
    if args.max_steps > 0:
        suffix += f"_bench{args.max_steps}"

    if args.accion == "train":
        ejecutar_entrenamiento(args, tokenizer, suffix, xpu_index, n_nodes, g_gpus)
    elif args.accion == "eval":
        ejecutar_evaluacion(args, tokenizer, suffix, xpu_index, n_nodes, g_gpus)
        
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
