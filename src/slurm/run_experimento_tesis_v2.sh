#!/bin/bash
#SBATCH --account=pci_146
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8

# === PARAMETRIZACIÓN ===
ACCION="train"
METHOD="desconocido"
PYTHON_ARGS=""

# Extraemos --accion para Bash y --method para el control de flujo condicional local.
# El resto de los parámetros se acoplan directamente a PYTHON_ARGS para ser resueltos
# de manera centralizada por el módulo argparse de Python.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --accion) 
            ACCION="$2"
            shift 2 
            ;;
        --method) 
            METHOD="$2"
            PYTHON_ARGS="$PYTHON_ARGS $1 $2"
            shift 2 
            ;;
        *) 
            PYTHON_ARGS="$PYTHON_ARGS $1"
            shift 1 
            ;;
    esac
done

MODEL_PATH="/home/mcamilo/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

module purge
module load intel/2025.3.0

CLUSTER_MKLROOT=$MKLROOT
eval "$(micromamba shell hook --shell bash)"
micromamba activate experimento

export MKLROOT=$CLUSTER_MKLROOT
export LD_PRELOAD=$CLUSTER_MKLROOT/lib/libmkl_intel_lp64.so:$CLUSTER_MKLROOT/lib/libmkl_gnu_thread.so:$CLUSTER_MKLROOT/lib/libmkl_core.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_lapack.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_blas.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_dft.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_sparse.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_vm.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_rng.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_stats.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_data_fitting.so

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

export PYTORCH_ENABLE_XPU_FALLBACK=1

echo ">>> Iniciando ($METHOD - $ACCION)..."
python experimento_sedici_tesis_v2.py --model_path "$MODEL_PATH" --accion "$ACCION" $PYTHON_ARGS
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: Falló la ejecución de $METHOD en fase $ACCION. Abortando."
    exit 1
fi

echo "=========================================================="
echo "Trabajo completado exitosamente."
echo "=========================================================="
