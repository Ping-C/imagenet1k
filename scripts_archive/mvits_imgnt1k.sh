#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=img1k                                # sets the job name if not set from environment
#SBATCH --array=3                       # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/mvits_1k_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/mvits_1k_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --ntasks=16
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lowpri
#SBATCH --nice=0                                              #positive means lower priority
#SBATCH --exclude=a100-st-p4d24xlarge-175





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
conda activate ffcv_v2

SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"



command_list[0]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_baseline.yaml --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.folder outputs/mvits_imgnt1k/baseline $SHARED_PARAM"
PYRAMID="--adv.radius_input=0.04 --adv.step_size_input=0.04  --adv.num_steps=1"
command_list[1]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_pyramid.yaml $PYRAMID --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.folder outputs/mvits_imgnt1k/pyramid_r0.04_s0.04_c1 $SHARED_PARAM"
PYRAMID="--adv.radius_input=0.04 --adv.step_size_input=0.01  --adv.num_steps=5"
command_list[2]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_pyramid.yaml $PYRAMID --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.folder outputs/mvits_imgnt1k/pyramid_r0.04_s0.01_c5 $SHARED_PARAM"

CACHE="--adv.radius_input=0.06 --adv.step_size_input=0.012 --adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
command_list[3]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_pyramid.yaml $CACHE --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.folder outputs/mvits_imgnt1k/pyramidcache_r0.06_s0.012 $SHARED_PARAM"



# jobs for adversarial finetuning
cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
# eval "$cur_command"
eval "srun -n $SLURM_NTASKS $cur_command"

