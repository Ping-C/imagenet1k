#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=train                                # sets the job name if not set from environment
#SBATCH --array=22                       # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/imgnt1k_vitb_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/imgnt1k_vitb_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --ntasks=32
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowpri
#SBATCH --exclude=a100-st-p4d24xlarge-121,a100-st-p4d24xlarge-278,a100-st-p4d24xlarge-158,a100-st-p4d24xlarge-271
#SBATCH --nice=0                                              #positive means lower priority
## SBATCH --dependency=                                              #positive means lower priority


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


command_list[0]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.split_backward=0 --training.batch_size=64 --adv.radius_input=0.05  --adv.step_size_input=0.05  --adv.adv_cache=1 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.05_cache --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[1]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.split_backward=0 --training.batch_size=64 --adv.radius_input=0.10  --adv.step_size_input=0.10  --adv.adv_cache=1 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.10_cache --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[2]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.split_backward=0 --training.batch_size=64 --adv.radius_input=0.15  --adv.step_size_input=0.15  --adv.adv_cache=1 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.20_cache --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[3]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.split_backward=0 --training.batch_size=64 --adv.radius_input=0.001 --adv.step_size_input=0.001 --adv.adv_cache=1 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.001_cache --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[4]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.split_backward=0 --training.batch_size=64 --adv.radius_input=0.002 --adv.step_size_input=0.002 --adv.adv_cache=1 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.002_cache --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[5]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.split_backward=0 --training.batch_size=64 --adv.radius_input=0.005 --adv.step_size_input=0.005 --adv.adv_cache=1 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.005_cache --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[6]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.split_backward=0 --training.batch_size=64 --adv.radius_input=0.010 --adv.step_size_input=0.010 --adv.adv_cache=1 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.010_cache --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[7]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.split_backward=0 --training.batch_size=64 --adv.radius_input=0.020 --adv.step_size_input=0.020 --adv.adv_cache=1 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.020_cache --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[8]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.002_cache/71647816-2d86-43a6-a517-c427a99bd7c4/params.json  --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[9]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.05_cache/28430aaa-ee8e-47dd-98ab-f96da9d618a3/params.json  --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[10]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.010_cache/cdb1f3b7-3495-4087-9c36-b8403b4658af/params.json  --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[11]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.20_cache/ae31717c-65a8-4f1b-82ff-4cef059fb916/params.json  --dist.port=$MASTER_PORT --dist.address=$master_addr"


SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[20]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml                                                                                                                                                      --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300aa --logging.folder output/pyramid/vitb_baseline $SHARED_PARAM"
command_list[21]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.2 --adv.step_size_input=0.02 --adv.adv_cache=1                                                                            --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300ab --logging.folder output/pyramid/vitb_r0.2_s0.02_cache $SHARED_PARAM"
command_list[22]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.4 --adv.step_size_input=0.04 --adv.adv_cache=1                                                                            --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300ac --logging.folder output/pyramid/vitb_r0.4_s0.04_cache $SHARED_PARAM"
command_list[23]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.6 --adv.step_size_input=0.06 --adv.adv_cache=1                                                                            --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300ad --logging.folder output/pyramid/vitb_r0.6_s0.06_cache $SHARED_PARAM"
command_list[24]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.2 --adv.step_size_input=0.02 --adv.adv_cache=1 --adv.cache_class_wise=1                                                   --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300aev2 --logging.folder output/pyramid/vitb_r0.2_s0.02_cacheclasswise $SHARED_PARAM"
command_list[25]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.01 --adv.step_size_input=0.01                                                                                             --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300af --logging.folder output/pyramid/vitb_r0.01_s0.01_c1 $SHARED_PARAM"
command_list[26]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.2 --adv.step_size_input=0.02 --adv.adv_cache=1 --adv.radius_schedule=1 --adv.max_radius_multiplier=2 --adv.start_epoch=30 --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300ag --logging.folder output/pyramid/vitb_r0.2_s0.02_cache_2-at-ep30 $SHARED_PARAM"

cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "srun -n $SLURM_NTASKS $cur_command"