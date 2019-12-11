#!/bin/bash
#SBATCH --job-name="test_job_mirroredstrategy"
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

python3 tf_keras_mirroredstrategy_mnist.py
