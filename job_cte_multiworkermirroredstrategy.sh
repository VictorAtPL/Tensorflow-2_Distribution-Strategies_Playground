#!/bin/bash
#SBATCH --job-name="test_job_multiworkermirroredstrategy"
#SBATCH --time=00-00:05:00
#SBATCH --workdir=.
#SBATCH --error=logs/error.log
#SBATCH --output=logs/output.log
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=160
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

source pc1_load_modules.sh

#export TF_CPP_MIN_LOG_LEVEL=0
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_SHM_DISABLE=0
#export NCCL_P2P_DISABLE=1

srun python3 tf_keras_multiworkermirroredstrategy_mnist.py
