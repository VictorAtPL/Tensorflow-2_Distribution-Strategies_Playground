#!/bin/bash
#SBATCH --job-name="tf_distribute_multiworkermirroredstategy"
#SBATCH --time=00-01:00:00
#SBATCH --workdir=.
#SBATCH --error=logs/multiworkermirroredstrategy_%j_error.log
#SBATCH --output=logs/multiworkermirroredstrategy_%j_output.log
#SBATCH --nodes=9
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=160
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive

source pc1_load_modules.sh

#export TF_CPP_MIN_LOG_LEVEL=0
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_SHM_DISABLE=0
#export NCCL_P2P_DISABLE=1

srun python3 tf_keras_multiworkermirroredstrategy.py "$@"
