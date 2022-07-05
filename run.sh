# export CUDA_VISIBLE_DEVICES=0
# python train_imagenet.py --config-file rn18_configs/rn18_16_epochs.yaml \
#     --data.train_dataset=/scratch/imagenet_ffcv/train_400_0.1_90.ffcv \
#     --data.val_dataset=/scratch/imagenet_ffcv/val_400_0.1_90.ffcv \
#     --logging.folder=./log_test

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
# python train_imagenet.py --config-file rn50_configs/rn50_16_epochs.yaml \
#     --data.train_dataset=/scratch/imagenet_ffcv/train_400_0.1_90.ffcv \
#     --data.val_dataset=/scratch/imagenet_ffcv/val_400_0.1_90.ffcv \
#     --logging.folder=./log_test

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
# python train_imagenet.py --config-file vit_configs/vit.yaml \
#     --data.train_dataset=/fsx/pingchiang/imagenet_ffcv_100class/train_400_0.1_90.ffcv \
#     --data.val_dataset=/fsx/pingchiang/imagenet_ffcv_100class/val_400_0.1_90.ffcv \
#     --logging.folder=./log_test \
#     --logging.log_level=2

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
# g
# python train_imagenet.py --config-file configs/mvit_1000cls.yaml
# python train_imagenet.py --config-file configs/vit_100cls.yaml
# python train_imagenet.py --config-file configs/vit_1000cls.yaml

# export IMAGENET_DIR='/datasets01/imagenet_full_size/061417'
# export WRITE_DIR='/data/home/pingchiang/data/imagenet_ffcv'
# bash write_imagenet.sh 500 0.50 90

# bash slurm_train.sh a100 imgnt --config-file configs/mvit_100cls.yaml &> logs/mvit100.log &
# bash slurm_train.sh a100 imgnt --config-file configs/mvit_1000cls.yaml &> logs/mvit1000.log &
# bash slurm_train.sh a100 imgnt --config-file configs/vit_100cls.yaml &> logs/vit100.log &
# bash slurm_train.sh a100 imgnt --config-file configs/vit_1000cls.yaml &> logs/vit1000.log &

# for r in 0.001 0.002 0.005 0.01 0.02 0.05 0.1
# do
# bash slurm_train.sh a100 imgnt --config-file configs/vit_100cls_advinput.yaml --adv.radius_input=$r --adv.step_size_input=$r --logging.folder=./outputs/vit_imagenet100_advinput$r &> logs/vit100_advinput${r}.log &
# done

# for r in 0.001 0.002 0.005 0.01 0.02 0.05
# do
# for l in 0 1 2 3 4 5
# do
# adv_features='{"'"$l"'":{"radius":'"$r"',"step_size":'"$r"'}}'
# bash slurm_train.sh a100 imgnt --config-file configs/vit_100cls_advfeature.yaml --adv.adv_features=$adv_features --logging.folder=./outputs/vit_imagenet100_advl${l}-${r}  &> logs/vit100_advl${l}-${r}.log &
# done
# done

# for r in 0.001 0.002 0.005 0.01 0.02
# do
# bash slurm_train.sh a100 sam --config-file configs/vit_100cls_sam.yaml --sam.radius=$r --logging.folder=./outputs/vit_imagenet100_sam$r &> logs/vit100_sam${r}.log &
# bash slurm_train.sh a100 sam --config-file configs/vit_1000cls_sam.yaml --sam.radius=$r --logging.folder=./outputs/vit_imagenet1000_sam$r &> logs/vit1000_sam${r}.log &
# done

# bash slurm_train.sh a100 sam --config-file configs/vit_1000cls.yaml --training.epochs 300 --logging.folder=./outputs/vit_imagenet1000_e300 &> logs/vit1000_e300.log &
# bash slurm_train.sh a100 sam --config-file configs/vit_100cls.yaml --training.epochs 300 --logging.folder=./outputs/vit_imagenet100_e300 &> logs/vit100_e300.log &
# bash slurm_train.sh a100 sam --config-file configs/mvit_1000cls.yaml --training.epochs 300 --logging.folder=./outputs/mvit_imagenet1000_e300 &> logs/mvit1000_e300.log &
# bash slurm_train.sh a100 sam --config-file configs/mvit_100cls.yaml --training.epochs 300 --logging.folder=./outputs/mvit_imagenet100_e300 &> logs/mvit100_e300.log &
# for r in 0.001 0.01 0.2
# do
# bash slurm_train.sh a100 sam --config-file configs/vit_1000cls.yaml --sam.radius=$r --training.epochs 300 --logging.folder=./outputs/vit_imagenet1000_sam${r}_e300 &> logs/vit1000_sam${r}_e300.log &
# bash slurm_train.sh a100 sam --config-file configs/vit_100cls.yaml --sam.radius=$r --training.epochs 300 --logging.folder=./outputs/vit_imagenet100_sam${r}_e300 &> logs/vit100_sam${r}_e300.log &
# bash slurm_train.sh a100 sam --config-file configs/mvit_1000cls.yaml --sam.radius=$r --training.epochs 300 --logging.folder=./outputs/mvit_imagenet1000_sam${r}_e300 &> logs/mvit1000_sam${r}_e300.log &
# bash slurm_train.sh a100 sam --config-file configs/mvit_100cls.yaml --sam.radius=$r --training.epochs 300 --logging.folder=./outputs/mvit_imagenet100_sam${r}_e300 &> logs/mvit100_sam${r}_e300.log &
# done

# bash slurm_train.sh a100 sam --config-file configs/vit_1000cls.yaml --sam.radius=$r --training.epochs 300 --training.fixed_dropout=1 --logging.folder=./outputs/vit_imagenet1000_sam${r}_e300_fixd &> logs/vit1000_sam${r}_e300_fixd.log &
# bash slurm_train.sh a100 sam --config-file configs/vit_100cls.yaml --sam.radius=$r --training.epochs 300 --training.fixed_dropout=1 --logging.folder=./outputs/vit_imagenet100_sam${r}_e300_fixed &> logs/vit100_sam${r}_e300_fixd.log &


# baseline
# adversarial training
# sam training
# bash slurm_train.sh a100 base --config-file configs/vit_100cls.yaml --training.mixup=False --training.epochs 300 --logging.folder=./outputs/nomixup/vit_imagenet100 &> logs/vit100nomixup.log &
# r=0.2
# bash slurm_train.sh a100 sam --config-file configs/vit_100cls.yaml --training.mixup=False --training.epochs 300 --sam.radius=$r --logging.folder=./outputs/nomixup/vit_imagenet100_sam${r} &> logs/vit100nomixup_sam${r}.log &
# bash slurm_train.sh a100 advi --config-file configs/vit_100cls_advinput.yaml --training.mixup=False --training.epochs 300 --adv.radius=0.002 --logging.folder=./outputs/nomixup/vit_imagenet100_advinput0.002 &> logs/vit100nomixup_advinput0.002.log &
# r=0.001
# for l in 0 1 2 3 4 5 6 7 8 9 10
# do
# adv_features='{"'"$l"'":{"radius":'"$r"',"step_size":'"$r"'}}'
# bash slurm_train.sh a100 advf --config-file configs/vit_100cls_advfeature.yaml --training.mixup=False --training.epochs 300  --adv.adv_features=$adv_features --logging.folder=./outputs/nomixup/vit_imagenet100_advfeature0.002_l$l &> logs/vit100nomixup_advfeature0.002_l${l}.log &
# done


# bash slurm_train.sh a100 base --config-file configs/vit_100cls.yaml --training.mixup=False --training.epochs 300 --logging.folder=./outputs/nomixup_v2/vit_imagenet100 &> logs/vit100nomixup_v2.log &
# r=0.2
# bash slurm_train.sh a100 sam --config-file configs/vit_100cls.yaml --training.mixup=False --training.epochs 300 --sam.radius=$r --logging.folder=./outputs/nomixup_v2/vit_imagenet100_sam${r} &> logs/vit100nomixup_v2_sam${r}.log &
# r=0.01
# bash slurm_train.sh a100 sam --config-file configs/vit_100cls.yaml --training.mixup=False --training.epochs 300 --sam.radius=$r --logging.folder=./outputs/nomixup_v2/vit_imagenet100_sam${r} &> logs/vit100nomixup_v2_sam${r}.log &

# for r in 0.001 0.0005 0.0002
# do
# bash slurm_train.sh a100 advi --config-file configs/vit_100cls_advinput.yaml --training.mixup=False --training.epochs 300 --adv.radius=$r --logging.folder=./outputs/nomixup_v2/vit_imagenet100_advinput$r &> logs/vit100nomixup_v2_advinput${r}.log &
# done

# for r in 0.001 0.0005
# do
# for l in 0 1 2 3 4 5 6 7 8 9 10
# do
# adv_features='{"'"$l"'":{"radius":'"$r"',"step_size":'"$r"'}}'
# bash slurm_train.sh a100 advf --config-file configs/vit_100cls_advfeature.yaml --training.mixup=False --training.epochs 300  --adv.adv_features=$adv_features --logging.folder=./outputs/nomixup_v2/vit_imagenet100_advfeature${r}_l$l &> logs/vit100nomixup_v2_advfeature${r}_l${l}.log &
# done
# done


bash slurm_train.sh a100 base --config-file configs/mvit_100cls.yaml --training.mixup=False --training.epochs 300 --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100 &> logs/mvit100nomixup_v2.log &
r=0.2
bash slurm_train.sh a100 sam --config-file configs/mvit_100cls.yaml --training.mixup=False --training.epochs 300 --sam.radius=$r --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100_sam${r} &> logs/mvit100nomixup_v2_sam${r}.log &
r=0.01
bash slurm_train.sh a100 sam --config-file configs/mvit_100cls.yaml --training.mixup=False --training.epochs 300 --sam.radius=$r --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100_sam${r} &> logs/mvit100nomixup_v2_sam${r}.log &

for r in 0.001 0.0005 0.0002
do
bash slurm_train.sh a100 advi --config-file configs/mvit_100cls_advinput.yaml --training.mixup=False --training.epochs 300 --adv.radius=$r --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100_advinput$r &> logs/mvit100nomixup_v2_advinput${r}.log &
done

# for r in 0.001 0.0005
# do
# for l in 0 1 2 3 4 5 6 7 8 9 10
# do
# adv_features='{"'"$l"'":{"radius":'"$r"',"step_size":'"$r"'}}'
# bash slurm_train.sh a100 advf --config-file configs/mvit_100cls_advfeature.yaml --training.mixup=False --training.epochs 300  --adv.adv_features=$adv_features --logging.folder=./outputs/nomixup_mvit/mvit_imagenet100_advfeature${r}_l$l &> logs/mvit100nomixup_v2_advfeature${r}_l${l}.log &
# done
# done