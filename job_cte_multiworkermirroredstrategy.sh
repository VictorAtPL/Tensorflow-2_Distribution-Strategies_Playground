#!/bin/bash
#SBATCH --job-name="test_job_multiworkermirroredstrategy"
#SBATCH --time=00-00:05:00
#SBATCH --workdir=.
#SBATCH --error=logs/error.log
#SBATCH --output=logs/output.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive

source pc1_load_modules.sh

srun python3 tf_keras_multiworkermirroredstrategy_mnist.py
