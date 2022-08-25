# Install Conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
eval "$(~/miniconda/bin/conda shell.bash hook)"
source ~/miniconda/bin/activate 
conda init
```

# Data Set Up
`cp -r /data/home/pingchiang/data/imagenet_ffcv /data/home/yipinzhou/`


# Environemnt Set Up - Option 1
```
conda config --env --set channel_priority flexible
conda create -y -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate ffcv
conda env config vars set NCCL_NSOCKS_PERTHREAD=4 NCCL_SOCKET_NTHREADS=2 NCC_INFO=INFO NCCL_SOCKET_IFNAME=ens32
pip install torchmetrics fvcore tensorflow tensorflow-datasets tensorflow_addons timm einops kornia wandb submitit
git clone https://github.com/ashertrockman/ffcv.git
cd ffcv
git checkout randaug
python setup.py install
```

# Environemnt Set Up - Option 2
```
source /data/home/pingchiang/miniconda/etc/profile.d/conda.sh
conda activate ffcv_v2
```

# WandB Setup

Username: pchiang

API key: 09e24c133b2a0e64eefa30ae9adb4dae08fe52cc

`wandb login 09e24c133b2a0e64eefa30ae9adb4dae08fe52cc`

# Test Script
Need to run this in an interactive shell
```
screen -R run
srun --gres=gpu:8 --cpus-per-gpu=8 --partition=hipri --account=all --pty /bin/bash -l
python train_imagenet_removehalf.py --config-file configs/simplevit/vits_1000cls.yaml --training.batch_size=128 --model.arch=vit_s_v7 --training.altnorm=1 --lr.warmup_epochs=8 --training.epochs=300 --logging.project_name test --training.weight_decay_explicit=1 --logging.resume_id=test --logging.folder outputs/test --dist.world_size=8 --dist.multinode=0 --data.loader_type=ffcv --training.mixup=0 --training.weight_decay_schedule=1
```

# Launch jobs with submitit
The following script automatically find nodes on the cluster and use any available fragmented resources at the time. It also relaunches the job when it times out and deals with node failures.
```
python -u launch.py --command="python train_imagenet.py --config-file configs/vit_100cls.yaml --training.epochs=3 --logging.folder outputs/test_success --logging.project_name test --dist.world_size=4 --data.num_workers=4" &> test1_success.log &
```