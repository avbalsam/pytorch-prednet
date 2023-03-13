#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -n 1 # number of tasks to be launched (can't be smaller than -N)
#SBATCH -c 4 # number of CPU cores associated with one GPU
#SBATCH --gres=gpu:1 # number of GPUs
##SBATCH --constraint=high-capacity
##SBATCH --constraint=32GB
#SBATCH --mem=64GB
#SBATCH --array=1
#SBATCH -D /om2/user/avbalsam/prednet/logs
cd /om2/user/avbalsam/prednet
hostname
echo "Plot Data"
date "+%y/%m/%d %H:%M:%S"

source /om2/user/jangh/miniconda/etc/profile.d/conda.sh
conda activate openmind
python plot_data.py