#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=img1k                                # sets the job name if not set from environment
#SBATCH --array=29                       # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/imgnt1k_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/imgnt1k_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --ntasks=16
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowpri
#SBATCH --nice=0                                              #positive means lower priority
###SBATCH --exclude=a100-st-p4d24xlarge-280,a100-st-p4d24xlarge-49





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

# jobs for requeueing over the weekend
command_list[8]="python train_imagenet.py --config-file outputs_requeue/imgnt1k/decoupled_0.001/954a9fc9-357f-4f6e-90c7-7f7ca8ebeda0/params.json --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[9]="python train_imagenet.py --config-file outputs_requeue/imgnt1k/decoupled_0.002/a0a9da2a-e3d4-4557-83ec-e85ad3988c34/params.json --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[10]="python train_imagenet.py --config-file outputs_requeue/imgnt1k/decoupled_0.005/cc39b35b-3a7f-4e08-a1f8-1f6c4179a48a/params.json --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[11]="python train_imagenet.py --config-file outputs_requeue/imgnt1k/decoupled_0.010/46f4e6a8-3e4b-4240-ae6d-2addc607761d/params.json --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[12]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.05 --adv.step_size_input 0.05 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.05 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[13]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.10 --adv.step_size_input 0.10 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.10 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[14]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.15 --adv.step_size_input 0.15 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.20 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[15]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.001 --adv.step_size_input 0.001 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.001 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[16]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.002 --adv.step_size_input 0.002 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.002 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[17]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.005 --adv.step_size_input 0.005 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.005 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[18]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --adv.radius_input 0.010 --adv.step_size_input 0.010 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_0.010 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[19]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.001/3f000a6b-941c-45d6-a752-c8e394ec37b6/params.json  --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[20]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.002/08a5e475-c316-4f72-98db-649c6620c669/params.json  --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[21]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.05/c13f78b2-8de9-4456-a24d-e0934d45c3a9/params.json   --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[22]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.005/160028eb-395f-428c-a58a-499e1848fe70/params.json  --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[23]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.10/915ff75e-9247-460c-b23a-99386860e524/params.json   --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[24]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.010/7949cf27-a6c7-49c0-98b5-715d630c834c/params.json  --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[25]="python train_imagenet.py --config-file outputs_requeue_v2/imgnt1k/decoupled_0.20/0b080854-2e11-4ee9-8f9b-0698a7b63781/params.json   --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[26]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_baseline.yaml           --logging.project_name imgnt1K --logging.resume_id=aa                                                                                         --logging.folder outputs/imgnt1k/baseline                                --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[27]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --logging.project_name imgnt1K --logging.resume_id=ab --adv.radius_input 0.20 --adv.step_size_input 0.020 --adv.num_steps 1 --adv.adv_cache=1 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_r0.20_s0.020_cache --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[28]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --logging.project_name imgnt1K --logging.resume_id=ac --adv.radius_input 0.30 --adv.step_size_input 0.030 --adv.num_steps 1 --adv.adv_cache=1 --logging.folder outputs_requeue_v2/imgnt1k/decoupled_r0.30_s0.030_cache --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[29]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --logging.project_name imgnt1K --logging.resume_id=ad --adv.radius_input 0.02 --adv.step_size_input 0.007 --adv.num_steps 3                   --logging.folder outputs_requeue_v2/imgnt1k/decoupled_r0.02_s0.007c3     --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[30]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --logging.project_name imgnt1K --logging.resume_id=ae --adv.radius_input 0.02 --adv.step_size_input 0.012 --adv.num_steps 2                   --logging.folder outputs_requeue_v2/imgnt1k/decoupled_r0.02_s0.012c2     --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[31]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_decoupled.yaml --logging.project_name imgnt1K --logging.resume_id=af --adv.radius_input 0.02 --adv.step_size_input 0.020 --adv.num_steps 1                   --logging.folder outputs_requeue_v2/imgnt1k/decoupled_r0.02_s0.020c1     --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

# jobs for adversarial finetuning
cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "srun -n $SLURM_NTASKS $cur_command"