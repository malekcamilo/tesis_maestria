# Estrategias de Adaptación Eficiente para Modelos de Lenguaje Abiertos

> **Tesis de Maestría en Inteligencia de Datos Orientada a Big  Data**
> Universidad Nacional de La Plata (UNLP) · Facultad de Informática
> **Autor:** Malek Camilo
> **Directores:** Juan Manuel Fernández · Marcelo Errecalde
> **Año:** 2026

Repositorio con el código fuente, conjuntos de datos y métricas de los experimentos realizados en la supercomputadora Clementina XXI. Incluye la implementación de seis estrategias de adaptación eficiente (LoRA, DoRA, QLoRA, QDoRA, GaLore y GaLore rSVD) aplicadas al modelo LLaMA 3.1 8B Instruct para la clasificación automática de documentos del repositorio institucional SEDICI.

---

## Mapa del repositorio

```text
tesis_maestria/
├── README.md
├── dataset/                              # Conjuntos de datos particionados (JSONL)
├── metricas/                             # Métricas consolidadas de los resultados (CSV)
└── src/                                  # Código fuente organizado
    ├── preprocesamiento/                 # Scripts para transformar datos de SEDICI
    ├── experimentos/                     # Script Python para entrenamiento y evaluación
    ├── slurm/                            # Scripts de lanzamiento para Clementina
    ├── scraping/                         # Scripts de recolección de datos
    ├── embeddings/                       # Ejemplos del estado del arte de embeddings
    └── tokenizacion/                     # Ejemplos del estado del arte de tokenización
```

| Carpeta                  | Qué es                                                                                                                                |
| --------------------------| ---------------------------------------------------------------------------------------------------------------------------------------|
| [`dataset/`](dataset/)   | Contiene las particiones de los datos procesados del repositorio SEDICI: `train_tesis.jsonl`, `val_tesis.jsonl` y `test_tesis.jsonl`. |
| [`metricas/`](metricas/) | Archivos CSV con las métricas consolidadas (globales y por clases) obtenidas durante el entrenamiento y la evaluación.                |
| [`src/`](src/)           | Código fuente organizado en subdirectorios temáticos (`preprocesamiento/`, `experimentos/`, etc.).                                    |

---

## Código y Experimentos (`src/`)

El desarrollo se enfoca en el fine-tuning de modelos de lenguaje mediante técnicas como FFT, LoRA, DoRA, QLoRA, QDoRA y GaLore.

> **Modelos PEFT Disponibles:** Los adaptadores correspondientes a las mejores cuatro técnicas evaluadas (QLoRA r=128, LoRA r=128, QDoRA r=128 y DoRA r=32) se encuentran disponibles públicamente en: [https://huggingface.co/malekcamilo](https://huggingface.co/malekcamilo)

### Ejecución de Experimentos

Los experimentos en el cluster Clementina se ejecutan utilizando el script principal `experimento_sedici_tesis_v2.py` (ubicado en `src/experimentos/`). A continuación se exponen algunos ejemplos de los comandos de lanzamiento en SLURM (ubicados en `src/slurm/`):

#### Entrenamiento

```bash
# - FFT
sbatch --job-name=fft-42 run_experimento_tesis_v2.sh --accion train --method fft --seed 42

# - DoRA r=32 (attn)
sbatch --job-name=dora-32-attn-42 run_experimento_tesis_v2.sh --accion train --method dora --lora_r 32 --target_modules attn --seed 42

# - LoRA r=32 (all)
sbatch --job-name=lora-32-all-42 run_experimento_tesis_v2.sh --accion train --method lora --lora_r 32 --target_modules all --seed 42

# - GaLore SVD g=500 (all)
sbatch --time=28:00:00 --cpus-per-task=64 --job-name=svd-128-500-1 run_experimento_tesis_v2.sh --accion train --method galore --galore_rank 128 --update_proj_gap 500 --target_modules all --seed 42
```

#### Evaluación

```bash
# - FFT
sbatch --job-name=fft-42 run_evaluacion_tesis_v2.sh --accion eval --method fft --seed 42

# - DoRA r=32 (attn)
sbatch --job-name=dora-32-attn-42 run_evaluacion_tesis_v2.sh --accion eval --method dora --lora_r 32 --target_modules attn --seed 42

# - LoRA r=32 (all)
sbatch --job-name=lora-32-all-42 run_evaluacion_tesis_v2.sh --accion eval --method lora --lora_r 32 --target_modules all --seed 42

# - GaLore SVD g=500 (all)
sbatch --job-name=svd-128-500-42 run_evaluacion_tesis_v2.sh --accion eval --method galore --galore_rank 128 --update_proj_gap 500 --target_modules all --seed 42
```

> **Nota:** Internamente, los lanzadores (como `run_experimento_tesis_v2.sh`) deben invocar al script en Python `src/experimentos/experimento_sedici_tesis_v2.py` y pasarle estos mismos argumentos.

### Dependencias del Entorno (Clementina)

Para la ejecución en Intel XPU, se utilizaron las siguientes dependencias principales:

- **PyTorch (Intel XPU):** `torch==2.11.0+xpu`, `pytorch-triton-xpu==3.5.0`
- **Modelos y NLP:** `transformers==5.5.0`, `peft==0.18.1`, `trl==1.0.0`, `accelerate==1.13.0`
- **Cuantización y Optimizadores:** `bitsandbytes==0.49.2`, `galore-torch==1.0`, `q-galore-torch==1.0`
- **Datos:** `datasets==4.8.4`
