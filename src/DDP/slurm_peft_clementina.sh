#!/bin/bash
#SBATCH --account=pci_146
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8

# Uso: sbatch -J dora-r32 slurm_peft_clementina.sh --method dora --lora_r 32
# Variables opcionales:
#   TILE_EN_PLACA=0   tile 0 de cada placa (por defecto 1)
#   MONITOR_MEM=1     memoria de cada nodo cada 2 s en el .out
#   FI_PROVIDER=mlx   provider de red (por defecto psm3)
#   STEP_TIMEOUT_S=N  corta si un paso tarda más de N s (por defecto 600)
#   RING_ALLREDUCE=1  fuerza ring en el allreduce
#   MODULO_INTEL=0    sin module intel ni LD_PRELOAD de MKL

MODEL_PATH="/home/mcamilo/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# 1. Entorno: módulo intel y MKL del cluster (opcional) + micromamba.
MODULO_INTEL=${MODULO_INTEL:-1}
module purge
if [ "$MODULO_INTEL" = 1 ]; then
    module load intel/2025.3.0
    CLUSTER_MKLROOT=$MKLROOT
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate experimento
if [ "$MODULO_INTEL" = 1 ]; then
    export MKLROOT=$CLUSTER_MKLROOT
    export LD_PRELOAD=$CLUSTER_MKLROOT/lib/libmkl_intel_lp64.so:$CLUSTER_MKLROOT/lib/libmkl_gnu_thread.so:$CLUSTER_MKLROOT/lib/libmkl_core.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_lapack.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_blas.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_dft.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_sparse.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_vm.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_rng.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_stats.so:$CLUSTER_MKLROOT/lib/libmkl_sycl_data_fitting.so
fi
echo ">>> MODULO_INTEL=$MODULO_INTEL | MKLROOT=${MKLROOT:-<no>} | LD_PRELOAD=${LD_PRELOAD:+sí}"

# 2. CPU: 8 hilos por rank.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# 3. Red: InfiniBand vía psm3; ring en el broadcast inicial de DDP.
export FI_PROVIDER=${FI_PROVIDER:-psm3}
export CCL_BCAST_ALGO=ring
# Allreduce: algoritmo de oneCCL, salvo RING_ALLREDUCE=1.
if [ "${RING_ALLREDUCE:-0}" = 1 ]; then export CCL_ALLREDUCE_ALGO=ring; fi
export FI_LOG_LEVEL=warn
export CCL_LOG_LEVEL=warn
export I_MPI_DEBUG=4  # pinning de CPU/NIC al arrancar

# 4. XPU: un tile por placa asignada (FLAT: placa k = tiles 2k y 2k+1). Multi-nodo exige nodo completo.
TILE_EN_PLACA=${TILE_EN_PLACA:-1}
PLACAS=${GPU_DEVICE_ORDINAL:-$SLURM_JOB_GPUS}
if [ -z "$PLACAS" ]; then
    echo "ERROR: Slurm no informó GPUs asignadas (GPU_DEVICE_ORDINAL vacío)."
    exit 1
fi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=$(echo "$PLACAS" | tr ',' '\n' | awk -v t="$TILE_EN_PLACA" '{printf "%s%d", (NR > 1 ? "," : ""), 2 * $1 + t}')
N_PLACAS=$(echo "$PLACAS" | tr ',' '\n' | wc -l)
if [ "$N_PLACAS" -lt "$SLURM_NTASKS_PER_NODE" ] || { [ "$SLURM_NNODES" -gt 1 ] && [ "$N_PLACAS" -ne 4 ]; }; then
    echo "ERROR: placas asignadas ($PLACAS) incompatibles con $SLURM_NTASKS_PER_NODE tasks/nodo en $SLURM_NNODES nodos."
    exit 1
fi
echo ">>> Placas asignadas por Slurm: $PLACAS -> ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
export PYTORCH_ENABLE_XPU_FALLBACK=1

# 5. Rendezvous: rank 0 en el primer nodo.
NODOS=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(echo "$NODOS" | head -n 1)
export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-0} % 10000)))

if [ "${MONITOR_MEM:-0}" = 1 ]; then
    srun --overlap --nodes="$SLURM_NNODES" --ntasks-per-node=1 bash monitor_mem.sh &
    MONITOR_PID=$!
fi

echo ">>> FI_PROVIDER=$FI_PROVIDER | CCL_ALLREDUCE_ALGO=${CCL_ALLREDUCE_ALGO:-<default>} | CCL_BCAST_ALGO=$CCL_BCAST_ALGO"
echo ">>> Iniciando DDP via mpirun: nodes=$(echo $NODOS | tr ' ' ','), tasks_per_node=$SLURM_NTASKS_PER_NODE, master=$MASTER_ADDR:$MASTER_PORT"
mpirun -np $((SLURM_NNODES * SLURM_NTASKS_PER_NODE)) -ppn "$SLURM_NTASKS_PER_NODE" \
    python experimento_peft_multi.py --model_path "$MODEL_PATH" "$@"
STATUS=$?
[ -n "$MONITOR_PID" ] && kill "$MONITOR_PID" 2>/dev/null
if [ $STATUS -ne 0 ]; then
    echo "ERROR: Falló la ejecución (exit $STATUS)."
    exit $STATUS
fi

echo "=========================================================="
echo "Trabajo completado."
echo "=========================================================="
