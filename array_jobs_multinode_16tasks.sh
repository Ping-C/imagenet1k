#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=img1k                                # sets the job name if not set from environment
#SBATCH --array=50                       # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/simplevits_re_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/simplevits_re_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --ntasks=16
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lowpri
#SBATCH --nice=0                                              #positive means lower priority
#SBATCH --exclude=a100-st-p4d24xlarge-175,a100-st-p4d24xlarge-120,a100-st-p4d24xlarge-65





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



# testing simple vit with randaug + mixup without label smoothing with adam
SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
PYRAMID_PARAM="--training.mixup=1 --training.randaug=1 --training.randaug_num_ops=2 --training.randaug_magnitude=10 --training.mixed_precision=0 --training.batch_size=64 --data.num_workers=8 --lr.warmup_epochs=4 --model.arch=vit_s --training.label_smoothing=0 --training.optimizer=adam"
command_list[32]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vits1000epoch300bqv3 --logging.folder output/pyramid/simplevits_baseline_v4_fast $SHARED_PARAM"


# testing simple vit with randaug + mixup without label smoothing with adam
SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
PYRAMID_PARAM="--training.mixup=1 --training.randaug=1 --training.randaug_num_ops=2 --training.randaug_magnitude=10 --training.mixed_precision=0 --training.batch_size=64 --data.num_workers=8 --lr.warmup_epochs=4 --model.arch=vit_s --training.label_smoothing=0 --training.optimizer=adam --training.altnorm=1"
command_list[33]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vits1000epoch300brv3 --logging.folder output/pyramid/simplevits_baseline_v5_altnorm_fast $SHARED_PARAM"


# testing simple vit with randaug + mixup without label smoothing with adam
SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
PYRAMID_PARAM="--training.mixup=1 --training.randaug=1 --training.randaug_num_ops=2 --training.randaug_magnitude=10 --training.mixed_precision=0 --training.batch_size=64 --data.num_workers=8 --lr.warmup_epochs=4 --model.arch=vit_s --training.label_smoothing=0 --training.altnorm=1"
command_list[34]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vits1000epoch300bsv3 --logging.folder output/pyramid/simplevits_baseline_v6_noadam_fast $SHARED_PARAM"

# testing simple vit with randaug + mixup without label smoothing with adam
SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
PYRAMID_PARAM="--training.mixup=1 --training.randaug=1 --training.randaug_num_ops=2 --training.randaug_magnitude=10 --training.mixed_precision=0 --training.batch_size=64 --data.num_workers=8 --lr.warmup_epochs=4 --model.arch=vit_b\ --training.altnorm=1"
command_list[35]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vits1000epoch300btv3 --logging.folder output/pyramid/simplevitb_baseline_v1fast $SHARED_PARAM"

# testing simple vit with randaug + mixup without label smoothing with adam
SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
PYRAMID_PARAM="--training.mixup=1 --training.randaug=1 --training.randaug_num_ops=2 --training.randaug_magnitude=10 --training.mixed_precision=0 --training.batch_size=64 --data.num_workers=8 --lr.warmup_epochs=32 --model.arch=vit_s --training.label_smoothing=0 --training.optimizer=adam"
command_list[36]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vits1000epoch300buv3 --logging.folder output/pyramid/simplevits_baseline_v3_warmup32 $SHARED_PARAM"


# testing simple vit with randaug + mixup without label smoothing with adam
SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
PYRAMID_PARAM="--training.mixup=1 --training.randaug=1 --training.randaug_num_ops=2 --training.randaug_magnitude=10 --training.mixed_precision=0 --training.batch_size=64 --data.num_workers=8 --lr.warmup_epochs=32 --model.arch=vit_s"
command_list[37]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vits1000epoch300bvv3 --logging.folder output/pyramid/simplevits_baseline_v4_warmup32 $SHARED_PARAM"


# testing simple vit with randaug + mixup without label smoothing with adam
SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
PYRAMID_PARAM="--training.mixup=1 --training.randaug=1 --training.randaug_num_ops=2 --training.randaug_magnitude=10 --training.mixed_precision=0 --training.batch_size=64 --data.num_workers=8 --lr.warmup_epochs=32 --model.arch=vit_b"
command_list[38]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vits1000epoch300bwv3 --logging.folder output/pyramid/simplevitb_baseline_v5_warmup32 $SHARED_PARAM"

# testing simple vit with randaug + mixup without label smoothing with adam
export WDS_EPOCH=1
SIMPLE_PARAM="--training.mixup=1 --training.randaug=1 --training.randaug_num_ops=2 --training.randaug_magnitude=10 --training.mixed_precision=0 --training.batch_size=64 --data.num_workers=8 --lr.warmup_epochs=32"
command_list[39]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90ab --logging.folder outputs/simple_vits_tune/simplevits_v1 $SHARED_PARAM"
command_list[40]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM --training.optimizer=adamw                                                                                                              --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90ac --logging.folder outputs/simple_vits_tune/simplevits_v2 $SHARED_PARAM"
command_list[41]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM --training.optimizer=adamw --data.torch_loader=1                                                                                        --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90ad --logging.folder outputs/simple_vits_tune/simplevits_v3 $SHARED_PARAM"

SIMPLE_PARAM="--training.mixup=1 --training.randaug=1 --training.randaug_num_ops=2 --training.randaug_magnitude=10 --training.mixed_precision=0 --training.batch_size=64 --data.num_workers=8 --lr.warmup_epochs=33 --training.epochs=90"
command_list[42]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM                                                                                                               --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90ae --logging.folder outputs/simple_vits_tune/simplevits_v4_warmup33 $SHARED_PARAM"
command_list[43]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM --data.torch_loader=1                                                                                         --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90af --logging.folder outputs/simple_vits_tune/simplevits_v5_warmup33_loader $SHARED_PARAM"

command_list[44]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM                                                                                                               --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90ag --logging.folder outputs/simple_vits_tune/simplevits_v6_warmup8 $SHARED_PARAM"
command_list[45]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM --data.torch_loader=1                                                                                         --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90ah --logging.folder outputs/simple_vits_tune/simplevits_v7_warmup8_loader $SHARED_PARAM"

command_list[46]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM --model.arch=vit_s_v2                                                                                         --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90aj --logging.folder outputs/simple_vits_tune/simplevitsv2_v8_warmup8 $SHARED_PARAM"
command_list[47]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM --model.arch=vit_s_v3                                                                                         --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90ak --logging.folder outputs/simple_vits_tune/simplevitsv3_v9_warmup8 $SHARED_PARAM"
command_list[48]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM --model.arch=vit_s_v4                                                                                         --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90al --logging.folder outputs/simple_vits_tune/simplevitsv4_v10_warmup8 $SHARED_PARAM"

SIMPLE_PARAM="--training.mixup=1 --training.randaug=1 --training.randaug_num_ops=2 --training.randaug_magnitude=10 --training.mixed_precision=0 --training.batch_size=64 --data.num_workers=8 --lr.warmup_epochs=8 --training.epochs=90"
command_list[49]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM --model.arch=vit_s_v4 --training.altnorm=1                                                                    --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90am --logging.folder outputs/simple_vits_tune/simplevitsv4_v11_warmup8_altnorm $SHARED_PARAM"
command_list[50]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml $SIMPLE_PARAM --model.arch=vit_s_v5 --training.altnorm=1                                                                    --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90an --logging.folder outputs/simple_vits_tune/simplevitsv5_v12_warmup8_altnorm $SHARED_PARAM"


# python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml --training.batch_size=128 --model.arch=vit_s_v5 --training.altnorm=1 --lr.warmup_epochs=8 --training.epochs=90 --logging.project_name imgnt1K --logging.resume_id=simplevits1000epoch90an --logging.folder outputs/simple_vits_tune/simplevitsv5_v12_warmup8_altnorm --dist.world_size=8 --dist.multinode=0
# python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml --training.batch_size=128 --model.arch=vit_s_v5 --training.altnorm=1 --lr.warmup_epochs=8 --training.epochs=90 --logging.project_name imgnt1K --training.weight_decay_explicit=1 --logging.resume_id=simplevits1000epoch90ao --logging.folder outputs/simple_vits_tune/simplevitsv5_v13_warmup8_altnorm_exwd --dist.world_size=8 --dist.multinode=0
# python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml --training.batch_size=128 --model.arch=vit_s_v5 --training.altnorm=1 --lr.warmup_epochs=8 --training.epochs=90 --logging.project_name imgnt1K --training.weight_decay_explicit=1 --training.weight_decay_nolr=1 --logging.resume_id=simplevits1000epoch90ap --logging.folder outputs/simple_vits_tune/simplevitsv5_v14_warmup8_altnorm_exwdnolr --dist.world_size=8 --dist.multinode=0

cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "srun -n $SLURM_NTASKS $cur_command"

