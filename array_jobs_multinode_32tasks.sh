#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=train                                # sets the job name if not set from environment
#SBATCH --array=44-47                       # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/imgnt1k_vitb_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/imgnt1k_vitb_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --ntasks=32
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lowpri
#SBATCH --nice=0                                              #positive means lower priority
#SBATCH --use-min-nodes                                              #positive means lower priority
#SBATCH --nodes=5-10                                              #positive means lower priority
#SBATCH --exclude=
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
conda activate ffcv_v2


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
command_list[20]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml                                                                                                                                                      --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300aav3 --logging.folder output/pyramid/vitb_baseline $SHARED_PARAM"
command_list[21]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.2 --adv.step_size_input=0.02 --adv.adv_cache=1                                                                            --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300abv3 --logging.folder output/pyramid/vitb_r0.2_s0.02_cache $SHARED_PARAM"
command_list[22]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.4 --adv.step_size_input=0.04 --adv.adv_cache=1                                                                            --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300acv3 --logging.folder output/pyramid/vitb_r0.4_s0.04_cache $SHARED_PARAM"
command_list[23]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.6 --adv.step_size_input=0.06 --adv.adv_cache=1                                                                            --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300adv3 --logging.folder output/pyramid/vitb_r0.6_s0.06_cache $SHARED_PARAM"
command_list[24]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.2 --adv.step_size_input=0.02 --adv.adv_cache=1 --adv.cache_class_wise=1                                                   --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300aev3 --logging.folder output/pyramid/vitb_r0.2_s0.02_cacheclasswise $SHARED_PARAM"
command_list[25]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.01 --adv.step_size_input=0.01                                                                                             --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300afv3 --logging.folder output/pyramid/vitb_r0.01_s0.01_c1 $SHARED_PARAM"
command_list[26]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --adv.radius_input=0.2 --adv.step_size_input=0.02 --adv.adv_cache=1 --adv.radius_schedule=1 --radius.max_multiplier=2 --radius.schedule_type=linear_increase --radius.start_epoch=30 --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300agv3 --logging.folder output/pyramid/vitb_r0.2_s0.02_cache_2-at-ep30 $SHARED_PARAM"

command_list[27]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml --training.mixup=0 --training.randaug=1                                                                                                               --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300ahv3 --logging.folder output/pyramid/vitb_baseline_randaug $SHARED_PARAM"
command_list[28]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=0.2 --adv.step_size_input=0.02 --adv.adv_cache=1                                     --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300aiv3 --logging.folder output/pyramid/vitb_r0.2_s0.02_cache_randaug $SHARED_PARAM"
command_list[29]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=0.01 --adv.step_size_input=0.01                                                      --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300ajv3 --logging.folder output/pyramid/vitb_r0.01_s0.01_c1_randaug $SHARED_PARAM"
SCHEDULE="--adv.radius_schedule=1 --radius.max_multiplier=10 --radius.start_epoch=10 --radius.schedule_type=linear_increase "
command_list[30]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=0.2 --adv.step_size_input=0.02 --adv.adv_cache=1 $SCHEDULE                           --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300akv3 --logging.folder output/pyramid/vitb_r0.2_s0.02_cache_randaug_inc10 $SHARED_PARAM"

PYRAMID_PARAM="--training.mixup=0 --training.randaug=1 --training.mixed_precision=0 --training.label_smoothing=0 --lr.warmup_epochs=3"
command_list[31]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300alv3 --logging.folder output/pyramid/vitb_baseline_pyramid $SHARED_PARAM"
command_list[32]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml $PYRAMID_PARAM --adv.radius_input=1.00  --adv.step_size_input=0.05  --adv.adv_cache=1                                                            --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300amv3 --logging.folder output/pyramid/vitb_r1_s0.05_cache_pyramid $SHARED_PARAM"

PYRAMID_PARAM="--training.mixup=0 --training.randaug=1 --training.mixed_precision=0 --training.label_smoothing=0 --lr.warmup_epochs=4"
command_list[33]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300anv3 --logging.folder output/pyramid/vitb_baseline_pyramid_v2 $SHARED_PARAM"

PYRAMID_PARAM="--training.mixup=0 --training.randaug=1 --training.mixed_precision=0 --lr.warmup_epochs=4"
command_list[34]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300aov3 --logging.folder output/pyramid/vitb_baseline_pyramid_v3 $SHARED_PARAM"
command_list[35]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM  --adv.radius_input=1.00  --adv.step_size_input=0.05  --adv.adv_cache=1                                                                 --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300apv3 --logging.folder output/pyramid/vitb_r1_s0.05_cache_pyramid_v3 $SHARED_PARAM"

SCHEDULE="--adv.radius_schedule=1 --radius.max_multiplier=1.5 --radius.min_multiplier=0.5 --radius.period_count 10 --radius.start_epoch=10 --radius.schedule_type=wave "
command_list[36]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=1 --adv.step_size_input=0.05 --adv.adv_cache=1 $SCHEDULE                              --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300aqv3 --logging.folder output/pyramid/vitb_r1_s0.05_cache_randaug_wave1.5-0.5 $SHARED_PARAM"

PYRAMID_PARAM="--training.mixup=0 --training.randaug=1 --training.mixed_precision=0 --lr.warmup_epochs=4 --model.arch=regvit_b"
command_list[37]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300arv3 --logging.folder output/pyramid/regvitb_baseline $SHARED_PARAM"

PYRAMID_PARAM="--training.mixup=0 --training.randaug=1 --training.label_smoothing=0 --training.mixed_precision=0 --lr.warmup_epochs=4 --model.arch=regvit_b"
command_list[38]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300asv3 --logging.folder output/pyramid/regvitb_baseline_nosmooth $SHARED_PARAM"

SCHEDULE="--adv.radius_schedule=1 --radius.max_multiplier=1.2 --radius.min_multiplier=0.8 --radius.period_count 10 --radius.start_epoch=10 --radius.schedule_type=wave "
command_list[39]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=1 --adv.step_size_input=0.05 --adv.adv_cache=1 $SCHEDULE                              --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300atv3 --logging.folder output/pyramid/vitb_r1_s0.05_cache_randaug_wave1.2-0.8 $SHARED_PARAM"

SCHEDULE="--adv.radius_schedule=1 --radius.max_multiplier=1.5 --radius.start_epoch=30 --radius.schedule_type=linear_increase "
command_list[40]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=1 --adv.step_size_input=0.05 --adv.adv_cache=1 $SCHEDULE                              --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300auv3 --logging.folder output/pyramid/vitb_r1_s0.05_cache_randaug_increase1.5 $SHARED_PARAM"

SCHEDULE="--adv.radius_schedule=1 --radius.min_multiplier=0.5 --radius.start_epoch=30 --radius.schedule_type=linear_decrease "
command_list[41]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=1 --adv.step_size_input=0.05 --adv.adv_cache=1 $SCHEDULE                              --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300avv3 --logging.folder output/pyramid/vitb_r1_s0.05_cache_randaug_decrease0.5 $SHARED_PARAM"

SCHEDULE="--adv.radius_schedule=1 --radius.max_multiplier=2 --radius.start_epoch=30 --radius.schedule_type=linear_increase "
command_list[42]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=1 --adv.step_size_input=0.05 --adv.adv_cache=1 $SCHEDULE                              --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300awv3 --logging.folder output/pyramid/vitb_r1_s0.05_cache_randaug_increase2 $SHARED_PARAM"

SCHEDULE="--adv.radius_schedule=1 --radius.min_multiplier=0 --radius.start_epoch=30 --radius.schedule_type=linear_decrease "
command_list[43]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=1 --adv.step_size_input=0.05 --adv.adv_cache=1 $SCHEDULE                              --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300axv3 --logging.folder output/pyramid/vitb_r1_s0.05_cache_randaug_decrease0 $SHARED_PARAM"


SCHEDULE="--adv.radius_schedule=1 --radius.min_multiplier=0 --radius.start_epoch=30 --radius.schedule_type=linear_decrease "
command_list[44]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=1.2 --adv.step_size_input=0.06 --adv.adv_cache=1 $SCHEDULE                              --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300bav3 --logging.folder output/pyramid/vitb_r1.2_s0.05_cache_randaug_decrease0 $SHARED_PARAM"

SCHEDULE="--adv.radius_schedule=1 --radius.min_multiplier=0 --radius.start_epoch=30 --radius.schedule_type=linear_decrease "
command_list[45]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=0.8 --adv.step_size_input=0.04 --adv.adv_cache=1 $SCHEDULE                              --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300bbv3 --logging.folder output/pyramid/vitb_r0.8_s0.05_cache_randaug_decrease0 $SHARED_PARAM"

command_list[46]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=0.8 --adv.step_size_input=0.05 --adv.adv_cache=1                                        --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300bcv3 --logging.folder output/pyramid/vitb_r0.8_s0.05_cache_randaug $SHARED_PARAM"
command_list[47]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls_cache.yaml --training.mixup=0 --training.randaug=1 --adv.radius_input=0.9 --adv.step_size_input=0.05 --adv.adv_cache=1                                        --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300bdv3 --logging.folder output/pyramid/vitb_r0.9_s0.05_cache_randaug $SHARED_PARAM"

PYRAMID_PARAM="--training.mixup=0 --training.randaug=1 --training.label_smoothing=0 --training.mixed_precision=0 --lr.warmup_epochs=32 --model.arch=regvit_b --training.optimizer=adam"
command_list[48]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300bev3 --logging.folder output/pyramid/regvitb_baseline_v3_warmup32 $SHARED_PARAM"

PYRAMID_PARAM="--training.mixup=0 --training.randaug=1 --training.mixed_precision=0 --lr.warmup_epochs=32 --model.arch=regvit_b --training.optimizer=adamw"
command_list[49]="python train_imagenet.py --config-file configs/pyramid/vitb_1000cls.yaml $PYRAMID_PARAM                                                                                                                                         --logging.project_name imgnt1K --logging.resume_id=vitb1000epoch300bfv3 --logging.folder output/pyramid/regvitb_baseline_v4_warmup32 $SHARED_PARAM"

cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "srun -n $SLURM_NTASKS $cur_command"