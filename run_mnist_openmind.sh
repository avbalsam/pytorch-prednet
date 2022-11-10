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

epochs=50
timesteps=4
learning_rate=0.0001
class_weight=0.1
rec_weight=0.9
noise_amt=0.0

while getopts e:t:l:c:r:n: flag
do
    case "${flag}" in
        e) epochs=${OPTARG};;
        t) timesteps=${OPTARG};;
        l) learning_rate=${OPTARG};;
        c) class_weight=${OPTARG};;
        r) rec_weight=${OPTARG};;
        n) noise_amt=${OPTARG};;
        *) pass
    esac
done

# echo "Epochs: $epochs";
# echo "Timesteps: $timesteps";
# echo "Learning rate: $learning_rate";
# echo "Classification weight: $class_weight";
# echo "Reconstruction weight: $rec_weight";

# module load openmind/singularity/3.4.1
# module add openmind/cuda/11.3
# module add openmind/cudnn/11.5-v8.3.3.40
# source /home/jangh/.bashrc
# conda activate openmind
source /om2/user/jangh/miniconda/etc/profile.d/conda.sh
conda activate openmind
python mnist_train.py -e "$epochs" -t "$timesteps" -l "$learning_rate" -c "$class_weight" -r "$rec_weight" -n "$noise_amt"
# --is_slurm=True \
# --job=${SLURM_ARRAY_JOB_ID} \
# --id=${SLURM_ARRAY_TASK_ID} \