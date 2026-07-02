import argparse
import json
import logging
import time
import torch
import unicodedata

from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, BitsAndBytesConfig
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

try:
    from galore_torch import GaLoreAdamW
    import galore_torch.galore_projector as gp
except ImportError:
    logger.warning("galore-torch no está instalado. Los métodos galore no funcionarán.")

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

def ejecutar_entrenamiento(args: argparse.Namespace, tokenizer: PreTrainedTokenizer, suffix: str, xpu_index: int) -> None:
    logger.info("Procesando configuración inicial (Entrenamiento)...")

    datasets = load_dataset("json", data_files={
        "train": "train_tesis.jsonl",
        "val": "val_tesis.jsonl"
    })

    def formatting_prompts_func(example: Dict[str, Any]) -> str:
        return format_prompt(example, is_training=True, eos_token=tokenizer.eos_token)

    if args.method == "base":
        logger.warning("El método 'base' no requiere entrenamiento.")
        return

    device_map = {"": f"xpu:{xpu_index}"}
    device_name = torch.xpu.get_device_name(xpu_index)
    prop = torch.xpu.get_device_properties(xpu_index)
    vram_gb = prop.total_memory / (1024**3)

    logger.info(f"[TELEMETRÍA XPU Single-GPU] Dispositivo: xpu:{xpu_index} -> {device_name} | VRAM: {vram_gb:.1f} GB")

    # Pattern Matching (Python 3.10+)
    bnb_config = None
    match args.method:
        case "qlora" | "qdora":
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

    custom_optimizer = None
    tasa_aprendizaje = 2e-5 

    target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj"] if args.target_modules == "attn" else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    svd_timing = {"total_seconds": 0.0, "call_count": 0}
    _original_get_orthogonal = gp.GaLoreProjector.get_orthogonal_matrix if hasattr(gp, 'GaLoreProjector') else None

    try:
        match args.method:
            case "lora" | "dora" | "qlora" | "qdora":
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

            case "galore" | "galore_rsvd":
                tasa_aprendizaje = 2e-5
                galore_params = []
                handled_params: Set[torch.nn.Parameter] = set()

                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear) and any(t in name for t in target_modules_list):
                        galore_params.append({
                            'params': [module.weight],
                            'rank': args.galore_rank,
                            'update_proj_gap': args.update_proj_gap,
                            'scale': 2.0,
                            'proj_type': 'std'
                        })
                        handled_params.add(module.weight)
                        if hasattr(module, 'bias') and module.bias is not None:
                            handled_params.add(module.bias)

                other_params = [p for p in model.parameters() if p not in handled_params]
                if other_params:
                    galore_params.append({'params': other_params})

                custom_optimizer = GaLoreAdamW(galore_params, lr=tasa_aprendizaje)

                if args.method == "galore_rsvd":
                    logger.info("Aplicando Monkey-Patch para rSVD en GaLoreProjector (torch.svd_lowrank)...")
                    def _rsvd_get_orthogonal(self: Any, weights: Any, rank: int, type: str) -> Any:
                        t0 = time.time()
                        float_data = weights.data.float() if weights.data.dtype != torch.float32 else weights.data
                        U, S, V = torch.svd_lowrank(float_data, q=rank)

                        if type == 'right':
                            result = V.T.to(weights.data.dtype)
                        elif type == 'left':
                            result = U.to(weights.data.dtype)
                        else:
                            raise ValueError(f"type {type} not supported")

                        svd_timing["total_seconds"] += time.time() - t0
                        svd_timing["call_count"] += 1
                        return result
                    gp.GaLoreProjector.get_orthogonal_matrix = _rsvd_get_orthogonal

                else:
                    logger.info("Instrumentando SVD exacta con timing...")
                    def _timed_get_orthogonal(self: Any, weights: Any, rank: int, type: str) -> Any:
                        t0 = time.time()
                        result = _original_get_orthogonal(self, weights, rank, type)
                        svd_timing["total_seconds"] += time.time() - t0
                        svd_timing["call_count"] += 1
                        return result
                    gp.GaLoreProjector.get_orthogonal_matrix = _timed_get_orthogonal

        training_args = SFTConfig(
            output_dir=f"./resultados_{suffix}",
            learning_rate=tasa_aprendizaje,
            seed=args.seed,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            num_train_epochs=5,
            bf16=True,
            save_strategy="no",
            logging_strategy="epoch",
            eval_strategy="epoch",
            max_length=512
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
            args=training_args,
            formatting_func=formatting_prompts_func,
            optimizers=(custom_optimizer, None) if custom_optimizer else (None, None)
        )

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

        train_time = (time.time() - start_time) / 3600
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

        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        log_path = logs_dir / f"loss_{suffix}.json"
        with open(log_path, 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2)

        match args.method:
            case "lora" | "dora" | "qlora" | "qdora":
                trainer.model.save_pretrained(f"./resultados_{suffix}")
            case _:
                trainer.save_model(f"./resultados_{suffix}")
        
        result_path = logs_dir / f"metricas_train_{suffix}.json"
        
        final_train_loss = next((log.get("loss") for log in reversed(trainer.state.log_history) if "loss" in log), "N/A")
        final_eval_loss = next((log.get("eval_loss") for log in reversed(trainer.state.log_history) if "eval_loss" in log), "N/A")
        resultados_train = {
            "config": {
                "fase": "train",
                "method": args.method,
                "lora_r": args.lora_r if args.method in ["lora", "dora", "qlora", "qdora"] else None,
                "galore_rank": args.galore_rank if args.method in ["galore", "galore_rsvd"] else None,
                "update_proj_gap": args.update_proj_gap if args.method in ["galore", "galore_rsvd"] else None,
                "target_modules": args.target_modules,
                "seed": args.seed
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
                "final_eval_loss": round(final_eval_loss, 4) if isinstance(final_eval_loss, float) else final_eval_loss
            }
        }
        if args.method in ["galore", "galore_rsvd"]:
            resultados_train["metrics"]["svd_total_seconds"] = round(svd_timing["total_seconds"], 2)
            resultados_train["metrics"]["svd_call_count"] = svd_timing["call_count"]
            resultados_train["metrics"]["svd_avg_ms_per_call"] = round(
                (svd_timing["total_seconds"] / max(svd_timing["call_count"], 1)) * 1000, 2
            )
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
        logger.info(f"{'Train Loss:':<12} {t_loss_str}")
        logger.info(f"{'Eval Loss:':<12} {e_loss_str}")
        if args.method in ["galore", "galore_rsvd"]:
            logger.info(f"{'SVD Total:':<12} {svd_timing['total_seconds']:.2f} s ({svd_timing['call_count']} calls)")
        logger.info(f"{'Pesos:':<12} ./resultados_{suffix}")
        logger.info(f"{'Historial:':<12} {log_path}")
        logger.info(f"{'Métricas:':<12} {result_path}")
        logger.info("=" * 50 + "\n")

    finally:
        # Clausura segura de Monkey-Patching
        if args.method in ["galore", "galore_rsvd"] and _original_get_orthogonal is not None:
            gp.GaLoreProjector.get_orthogonal_matrix = _original_get_orthogonal
            logger.debug("Monkey-Patching de GaLore restaurado exitosamente.")

def ejecutar_evaluacion(args: argparse.Namespace, tokenizer: PreTrainedTokenizer, suffix: str, xpu_index: int) -> None:
    logger.info("Procesando configuración inicial (Evaluación)...")

    datasets = load_dataset("json", data_files={
        "test": "test_tesis.jsonl"
    })

    match args.method:
        case "base":
            logger.info("Evaluando modelo base sin fine-tuning...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                dtype=torch.bfloat16,
                device_map={"": f"xpu:{xpu_index}"}
            )
        case "lora" | "dora" | "qlora" | "qdora":
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
                device_map={"": f"xpu:{xpu_index}"},
                quantization_config=bnb_config
            )
            model = PeftModel.from_pretrained(base_model, f"./resultados_{suffix}")
        case "fft" | "galore" | "galore_rsvd":
            logger.info(f"Cargando modelo completo ({args.method.upper()}) desde ./resultados_{suffix}")
            model = AutoModelForCausalLM.from_pretrained(
                f"./resultados_{suffix}",
                dtype=torch.bfloat16,
                device_map={"": f"xpu:{xpu_index}"}
            )

    acc, macro_f1, em_global, em_per_class, report, inference_time, samples_per_sec, tokens_per_sec = evaluate_model(model, tokenizer, datasets["test"])

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
            "galore_rank": args.galore_rank if args.method in ["galore", "galore_rsvd"] else None,
            "update_proj_gap": args.update_proj_gap if args.method in ["galore", "galore_rsvd"] else None,
            "target_modules": args.target_modules,
            "seed": args.seed
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--accion", type=str, choices=["train", "eval"], default="train", help="Fase a ejecutar de manera aislada (por defecto: train).")
    parser.add_argument("--method", type=str, choices=["base", "fft", "lora", "dora", "qlora", "qdora", "galore", "galore_rsvd"], required=True)
    parser.add_argument("--lora_r", type=int, default=8, help="Rango de LoRA (alpha = 2*r)")
    parser.add_argument("--galore_rank", type=int, default=128, help="Rango de proyección para GaLore")
    parser.add_argument("--update_proj_gap", type=int, default=500, help="Frecuencia de actualización de subespacio para GaLore")
    parser.add_argument("--target_modules", type=str, default="all", choices=["all", "attn"], help="Módulos a intervenir: 'all' (completo) o 'attn' (solo atención)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad.")
    args = parser.parse_args()

    info_extra = ""
    match args.method:
        case "lora" | "dora" | "qlora" | "qdora":
            info_extra = f" | Rank={args.lora_r}"
        case "galore" | "galore_rsvd":
            info_extra = f" | Rank={args.galore_rank} | Gap={args.update_proj_gap}"

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

    match args.method:
        case "lora" | "dora" | "qlora" | "qdora":
            suffix = f"{args.method}_r{args.lora_r}_{args.target_modules}_s{args.seed}"
        case "galore" | "galore_rsvd":
            suffix = f"{args.method}_r{args.galore_rank}_gap{args.update_proj_gap}_{args.target_modules}_s{args.seed}"
        case _:
            suffix = f"{args.method}_s{args.seed}"

    if not torch.xpu.is_available():
        raise RuntimeError("XPU no disponible. Este script requiere Intel XPU.")
        
    xpu_index = 0
    torch.xpu.set_device(xpu_index)

    match args.accion:
        case "train":
            ejecutar_entrenamiento(args, tokenizer, suffix, xpu_index)
        case "eval":
            ejecutar_evaluacion(args, tokenizer, suffix, xpu_index)

if __name__ == "__main__":
    main()
