#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=rn50                              # sets the job name if not set from environment
#SBATCH --array=12-15                    # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/rn50_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/rn50_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --partition=lowpri
#SBATCH --nice=0                                              #positive means lower priority
##SBATCH --dependency=afterany:31767_90                                             #positive means lower priority





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

# test resnet50
command_list[0]="python train_imagenet.py --config-file configs/rn50/base.yaml --logging.folder outputs/resnet50/rn50_baseline  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[1]="python train_imagenet.py --config-file configs/rn50/adv.yaml --logging.folder outputs/resnet50/rn50_advprop_r1s0.25c5  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[2]="python train_imagenet.py --config-file configs/rn50/adv.yaml --adv.radius_input=0.05 --adv.step_size_input=0.005 --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r0.05s0.005_cachem4  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[3]="python train_imagenet.py --config-file configs/rn50/adv.yaml --adv.radius_input=0.10 --adv.step_size_input=0.010 --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r0.10s0.010_cachem4  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[4]="python train_imagenet.py --config-file configs/rn50/adv.yaml --adv.radius_input=0.20 --adv.step_size_input=0.020 --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r0.20s0.020_cachem4  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[5]="python train_imagenet.py --config-file configs/rn50/adv.yaml --adv.radius_input=0.50 --adv.step_size_input=0.050 --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r0.50s0.050_cachem4  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[6]="python train_imagenet.py --config-file configs/rn50/adv.yaml --adv.radius_input=1.00 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r1.00s0.100_cachem4  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[7]="python train_imagenet.py --config-file configs/rn50/adv.yaml --adv.radius_input=2.00 --adv.step_size_input=0.200 --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r2.00s0.200_cachem4  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[8]="python train_imagenet.py --config-file configs/rn50/adv.yaml  --adv.radius_input=$(bc -l <<<'1/(255*0.25)')   --adv.step_size_input=$(bc -l <<<'0.25/(255*0.25)') --logging.folder outputs/resnet50/rn50_advprop_r1-255s0.25-255c5  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[9]="python train_imagenet.py --config-file configs/rn50/adv.yaml --adv.radius_input=$(bc -l <<<'0.5/(255*0.25)') --adv.step_size_input=$(bc -l <<<'0.05/(255*0.25)') --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r0.5-255s0.05-255_cachem4  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[10]="python train_imagenet.py --config-file configs/rn50/adv.yaml  --adv.radius_input=$(bc -l <<<'1/(255*0.25)')   --adv.step_size_input=$(bc -l <<<'0.1/(255*0.25)')  --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r1-255s0.1-255_cachem4  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[11]="python train_imagenet.py --config-file configs/rn50/adv.yaml  --adv.radius_input=$(bc -l <<<'2/(255*0.25)')   --adv.step_size_input=$(bc -l <<<'0.2/(255*0.25)')  --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r2-255s0.2-255_cachem4  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[12]="python train_imagenet.py --config-file configs/rn50/adv.yaml  --adv.radius_input=$(bc -l <<<'1/(255*0.25)')   --adv.step_size_input=$(bc -l <<<'0.25/(255*0.25)') --adv.adv_loss_even=1 --logging.folder outputs/resnet50/rn50_advprop_r1-255s0.25-255c5_evenloss  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[13]="python train_imagenet.py --config-file configs/rn50/adv.yaml --adv.radius_input=$(bc -l <<<'0.5/(255*0.25)') --adv.step_size_input=$(bc -l <<<'0.05/(255*0.25)')  --adv.adv_loss_even=1 --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r0.5-255s0.05-255_cachem4_evenloss  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[14]="python train_imagenet.py --config-file configs/rn50/adv.yaml  --adv.radius_input=$(bc -l <<<'1/(255*0.25)')   --adv.step_size_input=$(bc -l <<<'0.1/(255*0.25)') --adv.adv_loss_even=1 --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r1-255s0.1-255_cachem4_evenloss  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[15]="python train_imagenet.py --config-file configs/rn50/adv.yaml  --adv.radius_input=$(bc -l <<<'2/(255*0.25)')   --adv.step_size_input=$(bc -l <<<'0.2/(255*0.25)') --adv.adv_loss_even=1 --adv.adv_cache=1 --adv.cache_size_multiplier=4 --logging.folder outputs/resnet50/rn50_r2-255s0.2-255_cachem4_evenloss  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "srun -n $SLURM_NTASKS $cur_command"