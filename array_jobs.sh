#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=train                                # sets the job name if not set from environment
#SBATCH --array=60-63                        # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/mvitmixup_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/mvitmixup_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=24:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --gres=gpu:8
#SBATCH --account=all
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=lowpri
#SBATCH --nice=0                                              #positive means lower priority



source ~/miniconda/etc/profile.d/conda.sh
conda activate ffcv

i=0
for r in 0.005 0.01 0.02
do
for l in 3 7 11 
do
if [ $SLURM_ARRAY_TASK_ID -eq $i ]
then
export MASTER_PORT=$((12000 + $RANDOM % 20000))
adv_features='{"'"$l"'":{"radius":'"$r"',"step_size":'"$r"'}}'
python train_imagenet.py --config-file configs/vit_100cls_advfeature.yaml --training.mixup=0 --training.epochs 300  --adv.adv_features=$adv_features --logging.folder=./outputs/nomixup_v3/vit_imagenet100_advfeature${r}_l$l --dist.port=$MASTER_PORT 
fi
i=$((i+1))
done
done

if [ $SLURM_ARRAY_TASK_ID -eq 9 ]
then
export MASTER_PORT=$((12000 + $RANDOM % 20000))
python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --logging.folder=./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze4 --dist.port=$MASTER_PORT 
fi


if [ $SLURM_ARRAY_TASK_ID -eq 10 ]
then
export MASTER_PORT=$((12000 + $RANDOM % 20000))
python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --adv.freeze_layers=5 --logging.folder=./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze5 --dist.port=$MASTER_PORT 
fi

if [ $SLURM_ARRAY_TASK_ID -eq 11 ]
then
export MASTER_PORT=$((12000 + $RANDOM % 20000))
python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --adv.freeze_layers=3 --logging.folder=./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze3 --dist.port=$MASTER_PORT 
fi

if [ $SLURM_ARRAY_TASK_ID -eq 12 ]
then
export MASTER_PORT=$((12000 + $RANDOM % 20000))
python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --adv.freeze_layers=6 --logging.folder=./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze6 --dist.port=$MASTER_PORT 
fi

if [ $SLURM_ARRAY_TASK_ID -eq 13 ]
then
export MASTER_PORT=$((12000 + $RANDOM % 20000))
python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --adv.freeze_layers=7 --logging.folder=./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze6 --dist.port=$MASTER_PORT 
fi


export MASTER_PORT=$((12000 + $RANDOM % 20000))
command_list[14]="python train_imagenet.py --config-file configs/mvit_100cls.yaml --training.mixup=0 --training.epochs 300 --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100 --dist.port=$MASTER_PORT" 
command_list[15]="python train_imagenet.py --config-file configs/mvit_100cls.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100_adv0.01 --dist.port=$MASTER_PORT "
command_list[16]="python train_imagenet.py --config-file configs/mvit_100cls.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --adv.freeze_layers=1 --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100_adv0.01_freeze1 --dist.port=$MASTER_PORT" 
command_list[17]="python train_imagenet.py --config-file configs/mvit_100cls.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --adv.freeze_layers=2 --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100_adv0.01_freeze2 --dist.port=$MASTER_PORT" 
command_list[18]="python train_imagenet.py --config-file configs/mvit_100cls.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --adv.freeze_layers=3 --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100_adv0.01_freeze3 --dist.port=$MASTER_PORT" 
command_list[19]="python train_imagenet.py --config-file configs/mvit_100cls.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --adv.freeze_layers=4 --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100_adv0.01_freeze4 --dist.port=$MASTER_PORT" 
command_list[20]="python train_imagenet.py --config-file configs/mvit_100cls.yaml --training.mixup=0 --training.epochs 300 --adv.radius=0.001 --adv.freeze_layers=5 --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100_adv0.01_freeze5 --dist.port=$MASTER_PORT" 

command_list[21]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_twohead.yaml --training.mixup=0 --training.epochs=300 --adv.radius=0.001 --adv.flip=0 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.001_freeze4_twohead --dist.port=$MASTER_PORT"
command_list[22]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_twohead.yaml --training.mixup=0 --training.epochs=300 --adv.radius=0.001 --adv.flip=1 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.001_freeze4_twohead_flip --dist.port=$MASTER_PORT"

command_list[23]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_twohead.yaml --training.mixup=0 --training.epochs=300 --adv.radius=0.001 --adv.flip=1 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.001_freeze4_twohead_flip_v2 --dist.port=$MASTER_PORT"

command_list[24]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_decoupled.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.001 --adv.step_size_input=0.001 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.001_decoupled --dist.port=$MASTER_PORT"
command_list[25]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_decoupled.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.002 --adv.step_size_input=0.002 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.002_decoupled --dist.port=$MASTER_PORT"
command_list[26]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_decoupled.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.005 --adv.step_size_input=0.005 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.005_decoupled --dist.port=$MASTER_PORT"
command_list[27]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_decoupled.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.010 --adv.step_size_input=0.010 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.010_decoupled --dist.port=$MASTER_PORT"
command_list[28]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_decoupled.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.020 --adv.step_size_input=0.020 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.020_decoupled --dist.port=$MASTER_PORT"

command_list[29]="python train_imagenet.py --config-file configs/vit_100cls_advinput_advlosssmooth.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.001 --adv.step_size_input=0.001 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.001_advlosssmooth --dist.port=$MASTER_PORT"
command_list[30]="python train_imagenet.py --config-file configs/vit_100cls_advinput_advlosssmooth.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.002 --adv.step_size_input=0.002 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.002_advlosssmooth --dist.port=$MASTER_PORT"
command_list[31]="python train_imagenet.py --config-file configs/vit_100cls_advinput_advlosssmooth.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.005 --adv.step_size_input=0.005 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.005_advlosssmooth --dist.port=$MASTER_PORT"
command_list[32]="python train_imagenet.py --config-file configs/vit_100cls_advinput_advlosssmooth.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.010 --adv.step_size_input=0.010 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.010_advlosssmooth --dist.port=$MASTER_PORT"
command_list[33]="python train_imagenet.py --config-file configs/vit_100cls_advinput_advlosssmooth.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.020 --adv.step_size_input=0.020 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.020_advlosssmooth --dist.port=$MASTER_PORT"

command_list[34]="python train_imagenet.py --config-file configs/vit_100cls_advinput_advlosssmooth.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.050 --adv.step_size_input=0.050 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.050_advlosssmooth --dist.port=$MASTER_PORT"
command_list[35]="python train_imagenet.py --config-file configs/vit_100cls_advinput_advlosssmooth.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.100_advlosssmooth --dist.port=$MASTER_PORT"
command_list[36]="python train_imagenet.py --config-file configs/vit_100cls_advinput_advlosssmooth.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.200 --adv.step_size_input=0.200 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.200_advlosssmooth --dist.port=$MASTER_PORT"

command_list[37]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_decoupled.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.050 --adv.step_size_input=0.050 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.050_decoupled --dist.port=$MASTER_PORT"
command_list[38]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_decoupled.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.100_decoupled --dist.port=$MASTER_PORT"
command_list[39]="python train_imagenet.py --config-file configs/vit_100cls_advinput_freeze_decoupled.yaml  --training.mixup=0 --training.epochs=300 --adv.radius_input=0.200 --adv.step_size_input=0.200 --logging.folder output/nomixup_v3/vit_imagenet100_advinput0.200_decoupled --dist.port=$MASTER_PORT"

#epoch vs performance
command_list[40]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_baseline.yaml --training.epochs=100 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_baseline_e100 --dist.port=$MASTER_PORT"
command_list[41]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_baseline.yaml --training.epochs=200 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_baseline_e200 --dist.port=$MASTER_PORT"
command_list[42]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_baseline.yaml --training.epochs=300 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_baseline_e200 --dist.port=$MASTER_PORT"

#regular adv training
command_list[43]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput.yaml --training.epochs=300 --adv.radius_input=0.010 --adv.step_size_input=0.010 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_advinput0.010 --dist.port=$MASTER_PORT"
command_list[44]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput.yaml --training.epochs=300 --adv.radius_input=0.020 --adv.step_size_input=0.020 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_advinput0.020 --dist.port=$MASTER_PORT"
command_list[45]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput.yaml --training.epochs=300 --adv.radius_input=0.050 --adv.step_size_input=0.050 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_advinput0.050 --dist.port=$MASTER_PORT"

command_list[46]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.epochs=300 --adv.radius_input=0.010 --adv.step_size_input=0.010 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_advinput0.010_decoupled --dist.port=$MASTER_PORT"
command_list[47]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.epochs=300 --adv.radius_input=0.020 --adv.step_size_input=0.020 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_advinput0.020_decoupled --dist.port=$MASTER_PORT"
command_list[48]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.epochs=300 --adv.radius_input=0.050 --adv.step_size_input=0.050 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_advinput0.050_decoupled --dist.port=$MASTER_PORT"
command_list[49]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_advinput0.100_decoupled --dist.port=$MASTER_PORT"
command_list[50]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.epochs=300 --adv.radius_input=0.200 --adv.step_size_input=0.200 --logging.folder outputs/mvit_decoupled_nomixup/mvit_imagenet100_advinput0.200_decoupled --dist.port=$MASTER_PORT"

#mixup trainings
#epoch vs performance
command_list[51]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_baseline.yaml --training.mixup=1 --training.epochs=300 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_baseline_e300 --dist.port=$MASTER_PORT"

#regular adv training
command_list[52]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.010 --adv.step_size_input=0.010 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.010 --dist.port=$MASTER_PORT"
command_list[53]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.020 --adv.step_size_input=0.020 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.020 --dist.port=$MASTER_PORT"
command_list[54]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.050 --adv.step_size_input=0.050 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.050 --dist.port=$MASTER_PORT"

command_list[55]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.010 --adv.step_size_input=0.010 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.010_decoupled --dist.port=$MASTER_PORT"
command_list[56]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.020 --adv.step_size_input=0.020 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.020_decoupled --dist.port=$MASTER_PORT"
command_list[57]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.050 --adv.step_size_input=0.050 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.050_decoupled --dist.port=$MASTER_PORT"
command_list[58]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.100_decoupled --dist.port=$MASTER_PORT"
command_list[59]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.200 --adv.step_size_input=0.200 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.200_decoupled --dist.port=$MASTER_PORT"

#epoch vs performance
command_list[60]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_100cls_baseline.yaml --training.mixup=1 --training.epochs=300 --logging.folder outputs/mvit_decoupled_mixup/mvits_imagenet100_baseline_e300 --dist.port=$MASTER_PORT"

command_list[61]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.050 --adv.step_size_input=0.050 --logging.folder outputs/mvits_decoupled_mixup/mvits_imagenet100_advinput0.050_decoupled --dist.port=$MASTER_PORT"
command_list[62]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --logging.folder outputs/mvits_decoupled_mixup/mvits_imagenet100_advinput0.100_decoupled --dist.port=$MASTER_PORT"
command_list[63]="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.150 --adv.step_size_input=0.150 --logging.folder outputs/mvits_decoupled_mixup/mvits_imagenet100_advinput0.150_decoupled --dist.port=$MASTER_PORT"


cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "$cur_command"