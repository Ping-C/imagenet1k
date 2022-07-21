#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=vitc                               # sets the job name if not set from environment
#SBATCH --array=86                     # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/vit_cache_hyper_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/vit_cache_hyper_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --partition=lowpri
#SBATCH --nice=0                                              #positive means lower priority
##SBATCH --dependency=afterany:25046_69                                             #positive means lower priority





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

command_list[0]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.radius_input=0.01 --adv.step_size_input=0.01 --logging.folder output/nomixup_v3/vit_advinput0.01_triple --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[1]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.radius_input=0.02 --adv.step_size_input=0.02 --logging.folder output/nomixup_v3/vit_advinput0.02_triple --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[2]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.radius_input=0.05 --adv.step_size_input=0.05 --logging.folder output/nomixup_v3/vit_advinput0.05_triple --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[3]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.radius_input=0.10 --adv.step_size_input=0.10 --logging.folder output/nomixup_v3/vit_advinput0.10_triple --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[4]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.radius_input=0.20 --adv.step_size_input=0.20 --logging.folder output/nomixup_v3/vit_advinput0.20_triple --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[5]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.radius_input=0.005 --adv.step_size_input=0.005 --logging.folder output/nomixup_v3/vit_advinput0.005_triple --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[6]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.radius_input=0.002 --adv.step_size_input=0.002 --logging.folder output/nomixup_v3/vit_advinput0.002_triple --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[7]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.radius_input=0.001 --adv.step_size_input=0.001 --logging.folder output/nomixup_v3/vit_advinput0.001_triple --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[10]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.double_adv=1 --adv.radius_input=0.01 --adv.step_size_input=0.01 --logging.folder output/nomixup_v3/vit_advinput0.01_triple_dbadv --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[11]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.double_adv=1 --adv.radius_input=0.02 --adv.step_size_input=0.02 --logging.folder output/nomixup_v3/vit_advinput0.02_triple_dbadv --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[12]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.double_adv=1 --adv.radius_input=0.05 --adv.step_size_input=0.05 --logging.folder output/nomixup_v3/vit_advinput0.05_triple_dbadv --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[13]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.double_adv=1 --adv.radius_input=0.10 --adv.step_size_input=0.10 --logging.folder output/nomixup_v3/vit_advinput0.10_triple_dbadv --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[14]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.double_adv=1 --adv.radius_input=0.20 --adv.step_size_input=0.20 --logging.folder output/nomixup_v3/vit_advinput0.20_triple_dbadv --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[15]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.double_adv=1 --adv.radius_input=0.005 --adv.step_size_input=0.005 --logging.folder output/nomixup_v3/vit_advinput0.005_triple_dbadv --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[16]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.double_adv=1 --adv.radius_input=0.002 --adv.step_size_input=0.002 --logging.folder output/nomixup_v3/vit_advinput0.002_triple_dbadv --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[17]="python train_imagenet.py --config-file configs/vit_100cls_advinput_triple.yaml --adv.double_adv=1 --adv.radius_input=0.001 --adv.step_size_input=0.001 --logging.folder output/nomixup_v3/vit_advinput0.001_triple_dbadv --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"


# run multiple finetuning hyperparameter sweep
template="python train_imagenet.py --config-file outputs/nomixup_v3/vit_imagenet100/ab0b15c5-fd26-4fb9-a9c9-6fe01b447b21/params.json"\
" --logging.resume_id=ab0b15c5-fd26-4fb9-a9c9-6fe01b447b21 --logging.resume_checkpoint outputs/nomixup_v3/vit_imagenet100/ab0b15c5-fd26-4fb9-a9c9-6fe01b447b21/epoch299.pt "\
"--dist.multinode=1 --dist.world_size=$SLURM_NTASKS --dist.address=$master_addr --dist.port=$MASTER_PORT --model.arch vit_s_decoupled --adv.adv_loss_weight=0.5 --adv.num_steps=1"\
" --adv.radius_input=0.1 --adv.step_size_input=0.1 --logging.convert=1  --lr.warmup_epochs 0 --data.num_workers=4 "

command_list[20]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.001_ep100_fz10 --lr.lr 0.001 --training.freeze_nonlayernorm_epochs 10 --training.epochs 100 "
command_list[21]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.001_ep100_fz20 --lr.lr 0.001 --training.freeze_nonlayernorm_epochs 20 --training.epochs 100 "
command_list[22]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.001_ep100_fz30 --lr.lr 0.001 --training.freeze_nonlayernorm_epochs 30 --training.epochs 100 "
command_list[23]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.001_ep100_fz50 --lr.lr 0.001 --training.freeze_nonlayernorm_epochs 50 --training.epochs 100 "

command_list[24]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0005_ep100_fz10 --lr.lr 0.0005 --training.freeze_nonlayernorm_epochs 10 --training.epochs 100 "
command_list[25]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0005_ep100_fz20 --lr.lr 0.0005 --training.freeze_nonlayernorm_epochs 20 --training.epochs 100 "
command_list[26]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0005_ep100_fz30 --lr.lr 0.0005 --training.freeze_nonlayernorm_epochs 30 --training.epochs 100 "
command_list[27]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0005_ep100_fz50 --lr.lr 0.0005 --training.freeze_nonlayernorm_epochs 50 --training.epochs 100 "

command_list[28]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0002_ep100_fz10 --lr.lr 0.0002 --training.freeze_nonlayernorm_epochs 10 --training.epochs 100 "
command_list[29]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0002_ep100_fz20 --lr.lr 0.0002 --training.freeze_nonlayernorm_epochs 20 --training.epochs 100 "
command_list[30]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0002_ep100_fz30 --lr.lr 0.0002 --training.freeze_nonlayernorm_epochs 30 --training.epochs 100 "
command_list[31]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0002_ep100_fz50 --lr.lr 0.0002 --training.freeze_nonlayernorm_epochs 50 --training.epochs 100 "

command_list[32]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.001_ep50_fz10 --lr.lr 0.001 --training.freeze_nonlayernorm_epochs 10 --training.epochs 50 "
command_list[33]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.001_ep50_fz20 --lr.lr 0.001 --training.freeze_nonlayernorm_epochs 20 --training.epochs 50 "
command_list[34]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.001_ep50_fz30 --lr.lr 0.001 --training.freeze_nonlayernorm_epochs 30 --training.epochs 50 "

command_list[35]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0005_ep50_fz10 --lr.lr 0.0005 --training.freeze_nonlayernorm_epochs 10 --training.epochs 50 "
command_list[36]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0005_ep50_fz20 --lr.lr 0.0005 --training.freeze_nonlayernorm_epochs 20 --training.epochs 50 "
command_list[37]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0005_ep50_fz30 --lr.lr 0.0005 --training.freeze_nonlayernorm_epochs 30 --training.epochs 50 "

command_list[38]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0002_ep50_fz10 --lr.lr 0.0002 --training.freeze_nonlayernorm_epochs 10 --training.epochs 50 "
command_list[39]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0002_ep50_fz20 --lr.lr 0.0002 --training.freeze_nonlayernorm_epochs 20 --training.epochs 50 "
command_list[40]="$template --logging.folder output_finetune/vit_advinput0.1ft_lr0.0002_ep50_fz30 --lr.lr 0.0002 --training.freeze_nonlayernorm_epochs 30 --training.epochs 50 "

command_list[41]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.001 --adv.step_size_input=0.001 --adv.adv_cache=1 --logging.folder output/nomixup_v3/vit_advinput0.001_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[42]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.002 --adv.step_size_input=0.002 --adv.adv_cache=1 --logging.folder output/nomixup_v3/vit_advinput0.002_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[43]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.005 --adv.step_size_input=0.005 --adv.adv_cache=1 --logging.folder output/nomixup_v3/vit_advinput0.005_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[44]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.010 --adv.step_size_input=0.010 --adv.adv_cache=1 --logging.folder output/nomixup_v3/vit_advinput0.010_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[45]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.020 --adv.step_size_input=0.020 --adv.adv_cache=1 --logging.folder output/nomixup_v3/vit_advinput0.020_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[46]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.050 --adv.step_size_input=0.050 --adv.adv_cache=1 --logging.folder output/nomixup_v3/vit_advinput0.050_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[47]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --logging.folder output/nomixup_v3/vit_advinput0.100_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[50]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.001 --adv.step_size_input=0.001 --adv.adv_cache=1 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.001_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[51]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.002 --adv.step_size_input=0.002 --adv.adv_cache=1 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.002_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[52]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.005 --adv.step_size_input=0.005 --adv.adv_cache=1 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.005_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[53]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.010 --adv.step_size_input=0.010 --adv.adv_cache=1 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.010_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[54]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.020 --adv.step_size_input=0.020 --adv.adv_cache=1 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.020_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[55]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.050 --adv.step_size_input=0.050 --adv.adv_cache=1 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.050_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[56]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.100_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[57]="python train_imagenet.py --config-file configs/mvit_decoupled/mvit_100cls_advinput_decoupled.yaml --training.mixup=1 --training.epochs=300 --adv.radius_input=0.200 --adv.step_size_input=0.200 --adv.adv_cache=1 --logging.folder outputs/mvit_decoupled_mixup/mvit_imagenet100_advinput0.200_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[58]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_frequency=2  --logging.folder output/nomixup_v3/vit_advinput0.100_decoupled_cache_freq2  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[59]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_frequency=5  --logging.folder output/nomixup_v3/vit_advinput0.100_decoupled_cache_freq5  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[60]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_frequency=10 --logging.folder output/nomixup_v3/vit_advinput0.100_decoupled_cache_freq10 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[61]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_frequency=20 --logging.folder output/nomixup_v3/vit_advinput0.100_decoupled_cache_freq20 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

# tweak learning rate
command_list[62]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --lr.lr=0.002 --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_frequency=2  --logging.folder output/nomixup_v3/ht/vit_advinput0.100_decoupled_cache_freq2_lr0.002  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[63]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --lr.lr=0.004 --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_frequency=2  --logging.folder output/nomixup_v3/ht/vit_advinput0.100_decoupled_cache_freq2_lr0.004  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[64]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --lr.lr=0.006 --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_frequency=2  --logging.folder output/nomixup_v3/ht/vit_advinput0.100_decoupled_cache_freq2_lr0.006  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

# tweak the cache size of the adversarial filters
command_list[65]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_frequency=2 --adv.cache_size_multiplier=0.5 --logging.folder output/nomixup_v3/ht/vit_advinput0.100_decoupled_cache_freq2_cm0.5  --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[66]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_frequency=2 --adv.cache_size_multiplier=2   --logging.folder output/nomixup_v3/ht/vit_advinput0.100_decoupled_cache_freq2_cm2    --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[67]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.100 --adv.adv_cache=1 --adv.cache_frequency=2 --adv.cache_size_multiplier=4   --logging.folder output/nomixup_v3/ht/vit_advinput0.100_decoupled_cache_freq2_cm4    --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

# test some larger radius
command_list[68]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.200 --adv.step_size_input=0.200 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput0.200_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[69]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.300 --adv.step_size_input=0.300 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput0.300_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[70]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.400 --adv.step_size_input=0.400 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput0.400_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

# test smaller step sizes at radius 0.1
command_list[71]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.050 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput0.100_decoupled_cache_s0.05 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[72]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.020 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput0.100_decoupled_cache_s0.02 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[73]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.100 --adv.step_size_input=0.010 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput0.100_decoupled_cache_s0.01 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

# test larger step size
command_list[74]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.500 --adv.step_size_input=0.500 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput0.500_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[75]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.600 --adv.step_size_input=0.600 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput0.600_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[76]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.700 --adv.step_size_input=0.700 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput0.700_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[77]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=0.800 --adv.step_size_input=0.800 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput0.800_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[78]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=1.000 --adv.step_size_input=1.000 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput1.000_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[79]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=1.500 --adv.step_size_input=1.500 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput1.500_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[80]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=2.000 --adv.step_size_input=2.000 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput2.000_decoupled_cache --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[81]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=1.000 --adv.step_size_input=0.500 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput1.000_decoupled_cache_s0.50 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[82]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=1.000 --adv.step_size_input=0.250 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput1.000_decoupled_cache_s0.25 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[83]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=1.000 --adv.step_size_input=0.100 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput1.000_decoupled_cache_s0.10 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"

command_list[84]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=1.000 --adv.step_size_input=0.050 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput1.000_decoupled_cache_s0.050 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[85]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=1.000 --adv.step_size_input=0.020 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput1.000_decoupled_cache_s0.020 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[86]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=1.000 --adv.step_size_input=0.010 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput1.000_decoupled_cache_s0.010 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"
command_list[87]="python train_imagenet.py --config-file configs/vit_100cls_advinput_decoupled.yaml --training.mixup=0 --training.epochs=300 --adv.radius_input=1.000 --adv.step_size_input=0.005 --adv.adv_cache=1 --logging.folder output/nomixup_v3/ht/vit_advinput1.000_decoupled_cache_s0.005 --data.num_workers=4 --dist.world_size=$SLURM_NTASKS --dist.multinode=1 --dist.port=$MASTER_PORT --dist.address=$master_addr"




cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "srun -n $SLURM_NTASKS $cur_command"