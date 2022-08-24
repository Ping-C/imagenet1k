#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=img1k                                # sets the job name if not set from environment
#SBATCH --array=2-3,7-9,12-13                       # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/simplevits_1k_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/simplevits_1k_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --partition=lowpri
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

DATA="--training.batch_size=128"
# SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr $DATA"
SHARED_PARAM="--dist.world_size=8 --dist.multinode=0 --data.num_workers=64 $DATA"

command_list[0]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml                                                                                         --model.arch=vit_b_v5           --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300ba --logging.folder outputs/simplevits_1K/baseline $SHARED_PARAM"
# adv auxiliary
command_list[1]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml  --adv.radius_input=0.005 --adv.step_size_input=0.005                          --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bb --logging.folder outputs/simplevits_1K/advinput_r0.005s0.005 $SHARED_PARAM"
command_list[2]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml  --adv.radius_input=0.010 --adv.step_size_input=0.010                          --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bc --logging.folder outputs/simplevits_1K/advinput_r0.010s0.010 $SHARED_PARAM"
command_list[3]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml  --adv.radius_input=0.020 --adv.step_size_input=0.020                          --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bd --logging.folder outputs/simplevits_1K/advinput_r0.020s0.020 $SHARED_PARAM"

# adv auxiliary with cache + decreasing schedule
CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
command_list[4]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.5 --adv.step_size_input=0.025                      --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300ce --logging.folder outputs/simplevits_1K/advinputcache_r0.50s0.025_decrease $SHARED_PARAM"
command_list[5]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.6 --adv.step_size_input=0.03                       --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300cf --logging.folder outputs/simplevits_1K/advinputcache_r0.60s0.03_decrease $SHARED_PARAM"
command_list[6]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.4 --adv.step_size_input=0.02                       --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bg --logging.folder outputs/simplevits_1K/advinputcache_r0.40s0.02_decrease $SHARED_PARAM"
command_list[15]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.2 --adv.step_size_input=0.01                       --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bga --logging.folder outputs/simplevits_1K/advinputcache_r0.20s0.01_decrease $SHARED_PARAM"

# adv auxiliary with cache + no schedule
CACHE="--adv.adv_cache=1"
command_list[7]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.5   --adv.step_size_input=0.025                    --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bh --logging.folder outputs/simplevits_1K/advinputcache_r0.5s0.025 $SHARED_PARAM"
command_list[8]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.6   --adv.step_size_input=0.03                     --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bi --logging.folder outputs/simplevits_1K/advinputcache_r0.6s0.03  $SHARED_PARAM"
command_list[9]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.4   --adv.step_size_input=0.02                     --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bj --logging.folder outputs/simplevits_1K/advinputcache_r0.4s0.02  $SHARED_PARAM"
command_list[10]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.2   --adv.step_size_input=0.01                    --model.arch=vit_b_decoupled_v5 --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300ck --logging.folder outputs/simplevits_1K/advinputcache_r0.2s0.01  $SHARED_PARAM"

# advpyramid
command_list[11]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --adv.radius_input=0.04 --adv.step_size_input=0.04  --adv.num_steps=1        --model.arch=vit_b_v5           --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bl --logging.folder outputs/simplevits_1K/advinputpyramid_r0.04s0.04c1  $SHARED_PARAM"
command_list[12]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --adv.radius_input=0.04 --adv.step_size_input=0.02  --adv.num_steps=2        --model.arch=vit_b_v5           --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bm --logging.folder outputs/simplevits_1K/advinputpyramid_r0.04s0.02c2  $SHARED_PARAM"
command_list[13]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --adv.radius_input=0.04 --adv.step_size_input=0.015 --adv.num_steps=3        --model.arch=vit_b_v5           --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bn --logging.folder outputs/simplevits_1K/advinputpyramid_r0.04s0.015c3  $SHARED_PARAM"
command_list[14]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --adv.radius_input=0.04 --adv.step_size_input=0.01  --adv.num_steps=5        --model.arch=vit_b_v5           --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bo --logging.folder outputs/simplevits_1K/advinputpyramid_r0.04s0.01c5 $SHARED_PARAM"

#advcache pyramid
CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
command_list[15]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --adv.radius_input=0.06 --adv.step_size_input=0.012  $CACHE        --model.arch=vit_b_v5           --logging.project_name imgnt1K_simplevitb --logging.resume_id=simplevitb1000epoch300bp --logging.folder outputs/simplevits_1K/advinputpyramidcache_r0.06s0.012_decrease  $SHARED_PARAM"



# jobs for adversarial finetuning
cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "$cur_command"
# eval "srun -n $SLURM_NTASKS $cur_command"

