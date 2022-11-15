#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -n 1 # number of tasks to be launched (can't be smaller than -N)
#SBATCH -c 4 # number of CPU cores associated with one GPU
#SBATCH --gres=gpu:1 # number of GPUs
##SBATCH --constraint=high-capacity
##SBATCH --constraint=32GB
##SBATCH --mem=16GB
#SBATCH --array=1
#SBATCH -D /om2/user/avbalsam/prednet/logs
cd /om2/user/avbalsam/prednet
hostname
date "+%y/%m/%d %H:%M:%S"


while getopts v:s:m:d:n:b: flag
do
    case "${flag}" in
        v) version=${OPTARG};;
        s) is_slurm=${OPTARG};;
        m) model_name=${OPTARG};;
        d) data_name=${OPTARG};;
        n) noise=${OPTARG};;
        b) blur=${OPTARG};;
        *) pass
    esac
done

source /om2/user/jangh/miniconda/etc/profile.d/conda.sh
conda activate openmind
python mnist_train.py -v "$version" -s "$is_slurm" -m "$model_name" -d "$data_name" -n "$noise" -b "$blur"
# --is_slurm=True \
# --job=${SLURM_ARRAY_JOB_ID} \
# --id=${SLURM_ARRAY_TASK_ID} \