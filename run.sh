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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
python train_imagenet.py --config-file configs/mvit_100cls.yaml
python train_imagenet.py --config-file configs/mvit_1000cls.yaml
python train_imagenet.py --config-file configs/vit_100cls.yaml
python train_imagenet.py --config-file configs/vit_1000cls.yaml

# export IMAGENET_DIR='/datasets01/imagenet_full_size/061417'
# export WRITE_DIR='/data/home/pingchiang/data/imagenet_ffcv'
# bash write_imagenet.sh 500 0.50 90
