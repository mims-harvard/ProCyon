#!/bin/bash
#SBATCH -J tx_peptide
#SBATCH -o SAVEPATH/%x.%j.out
#SBATCH -e SAVEPATH/%x.%j.out
#SBATCH -p kempner_h100
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --constraint=h100
#SBATCH --mem=0
#SBATCH -t 0-24:00

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

base="HOME_DIR"
cd $base

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ib=( $( host $HOSTNAME | awk '{print $NF}' ) )
head_node_port=( $( comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1 ) )

#--mem-per-cpu=128G

echo Node IB: $heap_node_ib
export LOGLEVEL=INFO

export NCCL_DEBUG=INFO

export NCCL_SHM_DISABLE=0

# Define the SCRATCHDIR variable
SCRATCHDIR="{output_dir}"
# Define the NAME variable
NAME="{run_name}"
# Get the current date and time in the required format
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H:%M")
LOCALDIR="${CURRENT_DATETIME}_${NAME}"
# Export the SAVEDIR variable
export OUTPUTDIR="${SCRATCHDIR}/${LOCALDIR}"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL 

export MASTER_ADDR=$head_node_ib
export MASTER_PORT=$head_node_port

export WANDB__SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=1000

echo "Running:"

srun torchrun \
--nnodes=${SLURM_NNODES} \
--nproc-per-node=4 \
--max-restarts=0 \
--rdzv-id=456 \
--rdzv-backend=c10d \
--rdzv-endpoint=$head_node_ib:$head_node_port \
run_pretrain_IT_requeue.py \
--from_yaml configs/peptide_tune.yml