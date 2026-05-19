#!/bin/bash

#$ -cwd                      ## Execute a job in the current directory
#$ -l node_h=1               ## Use number of node
#$ -l h_rt=12:00:00          ## Running job time
#$ -j y                      ## Integrate standard error output into a standard output
#$ -p -5
#$ -m abe
#$ -M EMAIL_ADDRESS

config=${1:-configs/speechlm/default.yaml}
config_file=${2:-configs/speechlm/ddp.yaml}

module load cudnn/9.8.0
module load nccl/2.20.5
module load miniconda

main_process_ip=$(head -n 1 $PE_HOSTFILE | awk '{print $1}')

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate t4
accelerate launch \
    --config_file=${config_file} \
    --main_process_ip=${main_process_ip} \
    main_speechlm.py train \
    --config=${config}