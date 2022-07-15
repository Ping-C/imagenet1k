#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=train                                # sets the job name if not set from environment
#SBATCH --array=7                       # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/imgnt1k_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/imgnt1k_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --account=all
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowpri
#SBATCH --nice=0                                              #positive means lower priority
#SBATCH --exclude=a100-st-p4d24xlarge-280,a100-st-p4d24xlarge-49





module load /data/home/vkhalidov/modulefiles/cuda/11.3
module load /data/home/vkhalidov/modulefiles/nccl/2.12.7-cuda.11.3
module load /data/home/vkhalidov/modulefiles/nccl_efa/1.2.0-nccl.2.12.7-cuda.11.3

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCC_INFO=INFO
export NCCL_SOCKET_IFNAME=ens32

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$master_addr

export MASTER_PORT=$((12000 + $RANDOM % 20000))

source ~/miniconda/etc/profile.d/conda.sh
conda activate ffcv

command_list[0]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_baseline.yaml --logging.folder outputs/imgnt1k/baseline --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[1]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.05 --adv.step_size_input 0.05 --logging.folder outputs/imgnt1k/decoupled_0.05 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[2]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.10 --adv.step_size_input 0.10 --logging.folder outputs/imgnt1k/decoupled_0.10 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[3]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.15 --adv.step_size_input 0.15 --logging.folder outputs/imgnt1k/decoupled_0.20 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[4]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.001 --adv.step_size_input 0.001 --logging.folder outputs_requeue/imgnt1k/decoupled_0.001 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[5]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.002 --adv.step_size_input 0.002 --logging.folder outputs_requeue/imgnt1k/decoupled_0.002 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[6]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.005 --adv.step_size_input 0.005 --logging.folder outputs_requeue/imgnt1k/decoupled_0.005 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[7]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.010 --adv.step_size_input 0.010 --logging.folder outputs_requeue/imgnt1k/decoupled_0.010 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "srun -n $SLURM_NTASKS $cur_command"