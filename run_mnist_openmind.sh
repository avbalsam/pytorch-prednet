#!/bin/bash
SBATCH -t 01:30:00          # walltime = 1 hours and 30 minutes
SBATCH -n 1                 # one CPU (hyperthreaded) cores
hostname                     # Print the hostname of the compute node
# Execute commands to run your program here, taking Python for example,
python mnist_train.py