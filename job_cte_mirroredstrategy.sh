#!/bin/bash
#SBATCH --job-name="tf_distribute_mirroredstrategy"
#SBATCH --time=00-01:00:00
#SBATCH --workdir=.
#SBATCH --error=logs/mirroredstrategy_%j_error.log
#SBATCH --output=logs/mirroredstrategy_%j_output.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive

source pc1_load_modules.sh

#export TF_CPP_MIN_LOG_LEVEL=0
#export NCCL_DEBUG=INFO

python3 tf_keras_mirroredstrategy.py "$@"
