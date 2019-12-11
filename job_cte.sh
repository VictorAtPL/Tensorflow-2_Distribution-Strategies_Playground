#!/bin/bash
#SBATCH --job-name="test_job"
#SBATCH --qos=debug
#SBATCH --time=00-00:01:00
#SBATCH --workdir=.
#SBATCH --error=logs/error.log
#SBATCH --output=logs/output.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source pc1_load_modules.sh

python3 tf_keras_mnist.py
