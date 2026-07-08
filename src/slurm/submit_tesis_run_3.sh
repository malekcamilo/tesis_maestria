#!/bin/bash

echo "========================================"
echo "    LANZADOR DE EXPERIMENTOS TESIS      "
echo "========================================"

# Generar una única semilla a partir del OS
SEED=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')

echo "Semilla unificada generada: $SEED"
echo "Enviando trabajos a la cola de SLURM..."
echo "----------------------------------------"

# - FFT (paper)
sbatch --job-name=fft run_experimento_tesis_v2.sh --accion train --method fft --seed $SEED

# - LoRA / DoRA (attn)
sbatch --job-name=lora-8-attn run_experimento_tesis_v2.sh --accion train --method lora --lora_r 8 --target_modules attn --seed $SEED
sbatch --job-name=dora-8-attn run_experimento_tesis_v2.sh --accion train --method dora --lora_r 8 --target_modules attn --seed $SEED
sbatch --job-name=lora-16-attn run_experimento_tesis_v2.sh --accion train --method lora --lora_r 16 --target_modules attn --seed $SEED
sbatch --job-name=dora-16-attn run_experimento_tesis_v2.sh --accion train --method dora --lora_r 16 --target_modules attn --seed $SEED
sbatch --job-name=lora-32-attn run_experimento_tesis_v2.sh --accion train --method lora --lora_r 32 --target_modules attn --seed $SEED
sbatch --job-name=dora-32-attn run_experimento_tesis_v2.sh --accion train --method dora --lora_r 32 --target_modules attn --seed $SEED
sbatch --job-name=lora-64-attn run_experimento_tesis_v2.sh --accion train --method lora --lora_r 64 --target_modules attn --seed $SEED
sbatch --job-name=dora-64-attn run_experimento_tesis_v2.sh --accion train --method dora --lora_r 64 --target_modules attn --seed $SEED
sbatch --job-name=lora-128-attn run_experimento_tesis_v2.sh --accion train --method lora --lora_r 128 --target_modules attn --seed $SEED
sbatch --job-name=dora-128-attn run_experimento_tesis_v2.sh --accion train --method dora --lora_r 128 --target_modules attn --seed $SEED

# - LoRA, DoRA, QLoRA, QDoRA => r=8 (all)
sbatch --job-name=lora-8-all run_experimento_tesis_v2.sh --accion train --method lora --lora_r 8 --target_modules all --seed $SEED
sbatch --job-name=dora-8-all run_experimento_tesis_v2.sh --accion train --method dora --lora_r 8 --target_modules all --seed $SEED
sbatch --job-name=qlora-8-all run_experimento_tesis_v2.sh --accion train --method qlora --lora_r 8 --target_modules all --seed $SEED
sbatch --job-name=qdora-8-all run_experimento_tesis_v2.sh --accion train --method qdora --lora_r 8 --target_modules all --seed $SEED

# - LoRA, DoRA, QLoRA, QDoRA => r=16 (all)
sbatch --job-name=lora-16-all run_experimento_tesis_v2.sh --accion train --method lora --lora_r 16 --target_modules all --seed $SEED
sbatch --job-name=dora-16-all run_experimento_tesis_v2.sh --accion train --method dora --lora_r 16 --target_modules all --seed $SEED
sbatch --job-name=qlora-16-all run_experimento_tesis_v2.sh --accion train --method qlora --lora_r 16 --target_modules all --seed $SEED
sbatch --job-name=qdora-16-all run_experimento_tesis_v2.sh --accion train --method qdora --lora_r 16 --target_modules all --seed $SEED

# - LoRA, DoRA, QLoRA, QDoRA => r=32 (all)
sbatch --job-name=lora-32-all run_experimento_tesis_v2.sh --accion train --method lora --lora_r 32 --target_modules all --seed $SEED
sbatch --job-name=dora-32-all run_experimento_tesis_v2.sh --accion train --method dora --lora_r 32 --target_modules all --seed $SEED
sbatch --job-name=qlora-32-all run_experimento_tesis_v2.sh --accion train --method qlora --lora_r 32 --target_modules all --seed $SEED
sbatch --job-name=qdora-32-all run_experimento_tesis_v2.sh --accion train --method qdora --lora_r 32 --target_modules all --seed $SEED

# - LoRA, DoRA, QLoRA, QDoRA => r=64 (all)
sbatch --job-name=lora-64-all run_experimento_tesis_v2.sh --accion train --method lora --lora_r 64 --target_modules all --seed $SEED
sbatch --job-name=dora-64-all run_experimento_tesis_v2.sh --accion train --method dora --lora_r 64 --target_modules all --seed $SEED
sbatch --job-name=qlora-64-all run_experimento_tesis_v2.sh --accion train --method qlora --lora_r 64 --target_modules all --seed $SEED
sbatch --job-name=qdora-64-all run_experimento_tesis_v2.sh --accion train --method qdora --lora_r 64 --target_modules all --seed $SEED

# - LoRA, DoRA, QLoRA, QDoRA => r=128 (all)
sbatch --job-name=lora-128-all run_experimento_tesis_v2.sh --accion train --method lora --lora_r 128 --target_modules all --seed $SEED
sbatch --job-name=dora-128-all run_experimento_tesis_v2.sh --accion train --method dora --lora_r 128 --target_modules all --seed $SEED
sbatch --job-name=qlora-128-all run_experimento_tesis_v2.sh --accion train --method qlora --lora_r 128 --target_modules all --seed $SEED
sbatch --job-name=qdora-128-all run_experimento_tesis_v2.sh --accion train --method qdora --lora_r 128 --target_modules all --seed $SEED

# - GaLore rSVD => g=200 y g=500 (all)
sbatch --job-name=rsvd-128-200 run_experimento_tesis_v2.sh --accion train --method galore_rsvd --galore_rank 128 --update_proj_gap 200 --target_modules all --seed $SEED
sbatch --job-name=rsvd-128-500 run_experimento_tesis_v2.sh --accion train --method galore_rsvd --galore_rank 128 --update_proj_gap 500 --target_modules all --seed $SEED

# - GaLore SVD => g=200 y g=500 (all)
sbatch --time=28:00:00 --cpus-per-task=64 --job-name=svd-128-500 run_experimento_tesis_v2.sh --accion train --method galore --galore_rank 128 --update_proj_gap 500 --target_modules all --seed $SEED
