#!/bin/bash
#SBATCH --job-name="test_job"
#SBATCH --time=00-00:05:00
#SBATCH --workdir=.
#SBATCH --error=logs/nostrategy_%j_error.log
#SBATCH --output=logs/nostrategy_%j_output.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

source pc1_load_modules.sh

#export TF_CPP_MIN_LOG_LEVEL=0
#export NCCL_DEBUG=INFO

python3 tf_keras_mnist.py
