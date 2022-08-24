#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=img100                                # sets the job name if not set from environment
#SBATCH --array=58-62                       # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/simplevits_100_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/simplevits_100_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lowpri
#SBATCH --nice=0                                              #positive means lower priority
#SBATCH --exclude=a100-st-p4d24xlarge-129,a100-st-p4d24xlarge-120,a100-st-p4d24xlarge-265

# --gres=gpu:8 --nodes=1



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
if [ "$SLURM_NTASKS" -ne '1' ]; then
  SHARED_PARAM="--dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr $DATA"
else
  SHARED_PARAM="--dist.world_size=8 --dist.multinode=0 --data.num_workers=20 $DATA"
fi
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


CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
UNIV="--universal_feature_adv.step_size=0.001 --universal_feature_adv.radius=0.005 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[11]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cdl --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.005 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.002 --universal_feature_adv.radius=0.010 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[12]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cdm --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.010 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.004 --universal_feature_adv.radius=0.020 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[13]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cdn --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.020 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.008 --universal_feature_adv.radius=0.040 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[14]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cdo --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.040 $SHARED_PARAM"

UNIV="--universal_feature_adv.step_size=0.0005 --universal_feature_adv.radius=0.002 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[15]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cdp --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.002 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.0002 --universal_feature_adv.radius=0.001 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[16]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cdq --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.001 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.00002 --universal_feature_adv.radius=0.0001 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[17]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cdr --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.0001 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.00001 --universal_feature_adv.radius=0.00005 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[18]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cds --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.00005 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.00002 --universal_feature_adv.radius=0.0000 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[19]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cft --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.0000 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.00002 --universal_feature_adv.radius=0.0000 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[20]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=test1 --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.0000 --dist.world_size=0 --training.distributed=0 --dist.multinode=0 --data.num_workers=20 $DATA"
UNIV="--universal_feature_adv.step_size=0.00002 --universal_feature_adv.radius=0.0000 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_v5"
command_list[21]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300cdu --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.0000_oriarch $SHARED_PARAM"


UNIV="--universal_feature_adv.step_size=0.02 --universal_feature_adv.radius=0.100 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[22]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300av --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.10 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.04 --universal_feature_adv.radius=0.200 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[23]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300aw --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.20 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.08 --universal_feature_adv.radius=0.400 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[24]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300ax --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ0.40 $SHARED_PARAM"

UNIV="--universal_feature_adv.step_size=0.2 --universal_feature_adv.radius=1.00 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[25]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300da --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ1.00 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=0.4 --universal_feature_adv.radius=2.00 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[26]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300db --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ2.00 $SHARED_PARAM"
UNIV="--universal_feature_adv.step_size=1.0 --universal_feature_adv.radius=5.00 --universal_feature_adv.layers=0 --model.arch=vit_s_decoupled_universal_v5"
command_list[27]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300dc --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univ5.00 $SHARED_PARAM"

I=28
for R in 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
EXPRESSION="$R/5"
STEP_SIZE=$( bc -l <<< $EXPRESSION )
UNIV="--universal_feature_adv.step_size=$STEP_SIZE --universal_feature_adv.radius=$R --universal_feature_adv.layers=5 --model.arch=vit_s_decoupled_universal_v5"
command_list[I]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300dd$I --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univl5$R $SHARED_PARAM"
I=$((I+1))
UNIV="--universal_feature_adv.step_size=$STEP_SIZE --universal_feature_adv.radius=$R --universal_feature_adv.layers=11 --model.arch=vit_s_decoupled_universal_v5"
command_list[I]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300dd$I --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univl11$R $SHARED_PARAM"
I=$((I+1))
done
# I == 48
for R in 0.0005 0.001 0.002 0.005
do
EXPRESSION="$R/5"
STEP_SIZE=$( bc -l <<< $EXPRESSION )
UNIV="--universal_feature_adv.step_size=$STEP_SIZE --universal_feature_adv.radius=$R --universal_feature_adv.layers=0,1,2,3,4,5,6,7,8,9,10,11 --model.arch=vit_s_decoupled_universal_v5"
command_list[I]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300dd$I --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univall$R $SHARED_PARAM"
I=$((I+1))
done
# I == 52
CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0 --adv.pyramid=1"
command_list[52]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300de2 --logging.folder outputs/simplevits_100/advinputpyramidcache_r0.40s0.02_decrease $SHARED_PARAM"


CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0 --adv.pyramid=1"
command_list[53]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.1 --adv.step_size_input=0.005                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300df --logging.folder outputs/simplevits_100/advinputpyramidcache_r0.10s0.005_decrease $SHARED_PARAM"
CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0 --adv.pyramid=1"
command_list[54]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.05 --adv.step_size_input=0.0025                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300dg --logging.folder outputs/simplevits_100/advinputpyramidcache_r0.05s0.0025_decrease $SHARED_PARAM"
CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0 --adv.pyramid=1 --model.arch=vit_s_v5"
command_list[55]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.05 --adv.step_size_input=0.0025                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300dh --logging.folder outputs/simplevits_100/advinputpyramidcache_r0.05s0.0025_decrease_nodecoupled $SHARED_PARAM"

CACHE="--adv.adv_cache=1 --adv.radius_schedule=0 --adv.pyramid=1"
command_list[56]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.05 --adv.step_size_input=0.0025                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300di --logging.folder outputs/simplevits_100/advinputpyramidcache_r0.05s0.0025 $SHARED_PARAM"
CACHE="--adv.adv_cache=1 --adv.radius_schedule=0 --adv.pyramid=1 --model.arch=vit_s_v5"
command_list[57]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE --adv.radius_input=0.05 --adv.step_size_input=0.0025                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300dj --logging.folder outputs/simplevits_100/advinputpyramidcache_r0.05s0.0025_nodecoupled $SHARED_PARAM"

I=58
CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
for R in 0.01 0.02 0.05 0.1 0.2
do
EXPRESSION="$R/5"
STEP_SIZE=$( bc -l <<< $EXPRESSION )
UNIV="--universal_feature_adv.step_size=$STEP_SIZE --universal_feature_adv.radius=$R --universal_feature_adv.layers=0,1,2,3,4,5,6,7,8,9,10,11 --model.arch=vit_s_decoupled_universal_v5"
command_list[I]="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advinput.yaml $CACHE $UNIV --adv.radius_input=0.4 --adv.step_size_input=0.02                     --logging.project_name imgnt100_simplevits --logging.resume_id=simplevits1000epoch300dd$I --logging.folder outputs/simplevits_100/advinputcache_r0.40s0.02_decrease_univall$R $SHARED_PARAM"
I=$((I+1))
done

# jobs for adversarial finetuning
cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
if [ "$SLURM_NTASKS" -ne '1' ]; then
  echo "srun -n $SLURM_NTASKS $cur_command"
  eval "srun -n $SLURM_NTASKS $cur_command"
else
  echo $cur_command
  eval "$cur_command"
fi
# eval "srun $cur_command"
# eval "srun -n $SLURM_NTASKS $cur_command"

