# setting some environment variables for multinode jobs
source /data/home/pingchiang/miniconda/etc/profile.d/conda.sh
conda activate ffcv_v2

module load /data/home/vkhalidov/modulefiles/cuda/11.3
module load /data/home/vkhalidov/modulefiles/nccl/2.12.7-cuda.11.3
module load /data/home/vkhalidov/modulefiles/nccl_efa/1.2.0-nccl.2.12.7-cuda.11.3

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCC_INFO=INFO
export NCCL_SOCKET_IFNAME=ens32

# launch jobs
python -u launch.py --command="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_pyramid.yaml --adv.radius_input=0.04 --adv.step_size_input=0.04  --adv.num_steps=1 --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.folder outputs/mvits_imgnt1k/pyramid_r0.04_s0.04_c1 --dist.world_size=16" &> run1-1.log &
python -u launch.py --command="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_pyramid.yaml --adv.radius_input=0.04 --adv.step_size_input=0.01  --adv.num_steps=5 --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.folder outputs/mvits_imgnt1k/pyramid_r0.04_s0.01_c5 --dist.world_size=16" &> run1-2.log &