#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=img100                                # sets the job name if not set from environment
#SBATCH --array=6-9                       # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/simplevits_100_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/simplevits_100_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --partition=hipri
#SBATCH --nice=0                                              #positive means lower priority
#SBATCH --exclude=





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

DATA="--data.num_classes=100 --data.train_dataset=/fsx/pingchiang/imagenet_ffcv_100class/train_400_0.1_90.ffcv --data.val_dataset=/fsx/pingchiang/imagenet_ffcv_100class/val_400_0.1_90.ffcv"
# SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr $DATA"
SHARED_PARAM="--dist.world_size=8 --dist.multinode=0 --data.num_workers=20 $DATA"

command_list[0]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml                                                                                       --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300ba --logging.folder outputs/simplevits_100/baseline $SHARED_PARAM"
command_list[1]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml                                                                              --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300bb --logging.folder outputs/simplevits_100/advinput_r0.005s0.005 $SHARED_PARAM"
CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
command_list[2]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.5 --adv.step_size_input=0.025                    --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300bc --logging.folder outputs/simplevits_100/advinputcache_r0.50s0.025_decrease $SHARED_PARAM"
command_list[3]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.6 --adv.step_size_input=0.03                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300bd --logging.folder outputs/simplevits_100/advinputcache_r0.60s0.03_decrease $SHARED_PARAM"
command_list[4]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300be --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease $SHARED_PARAM"
CACHE="--adv.adv_cache=1"
command_list[5]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=1   --adv.step_size_input=0.05                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300bf --logging.folder outputs/simplevits_100/advinputcache_r1.00s0.05 $SHARED_PARAM"

command_list[6]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --adv.radius_input=0.03 --adv.step_size_input=0.03   --adv.num_steps=1      --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cg --logging.folder outputs/simplevits_100/advinputpyramid_r0.03s0.03c1  $SHARED_PARAM"
command_list[7]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --adv.radius_input=0.03 --adv.step_size_input=0.015  --adv.num_steps=2      --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300ch --logging.folder outputs/simplevits_100/advinputpyramid_r0.03s0.015c2  $SHARED_PARAM"
command_list[8]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --adv.radius_input=0.03 --adv.step_size_input=0.010  --adv.num_steps=3      --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300ci --logging.folder outputs/simplevits_100/advinputpyramid_r0.03s0.010c3  $SHARED_PARAM"
command_list[9]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --adv.radius_input=0.03 --adv.step_size_input=0.0075 --adv.num_steps=5      --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cj --logging.folder outputs/simplevits_100/advinputpyramid_r0.03s0.0075c5 $SHARED_PARAM"

CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
command_list[10]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.1   --adv.step_size_input=0.01                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300bk --logging.folder outputs/simplevits_100/advinputcache_r0.10s0.01 $SHARED_PARAM"


# jobs for adversarial finetuning
cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "srun $cur_command"
# eval "srun -n $SLURM_NTASKS $cur_command"

