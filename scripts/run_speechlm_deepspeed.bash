#!/bin/bash

#$ -cwd                      ## Execute a job in the current directory
#$ -l node_f=8               ## Use number of node
#$ -l h_rt=24:00:00          ## Running job time
#$ -j y                      ## Integrate standard error output into a standard output
#$ -p -5
#$ -m abe
#$ -M EMAIL_ADDRESS

config=${1:-configs/speechlm/default.yaml}
hostfile=${2:-configs/speechlm/hostfile}

module load openmpi/5.0.7-nvhpc
module load cudnn/9.0.0
module load nccl/2.20.5
module load miniconda

main_process_ip=$(head -n 1 $PE_HOSTFILE | awk '{print $1}')
awk '{print $1 " slots=4"}' "$PE_HOSTFILE" > $hostfile

export MASTER_ADDR=$main_process_ip
export MASTER_PORT=29500
rm .deepspeed_env

mpirun \
    --hostfile $hostfile \
    -npernode 4 \
    -n 32 \
    --bind-to none \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x LD_LIBRARY_PATH \
    bash -c '
    eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
    conda activate t4
    python main_speechlm.py train --config='"${config}"'
'