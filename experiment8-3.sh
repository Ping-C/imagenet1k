# setting some environment variables for multinode jobs
source ~/miniconda/etc/profile.d/conda.sh
conda activate ffcv_v2

module load /data/home/vkhalidov/modulefiles/cuda/11.3
module load /data/home/vkhalidov/modulefiles/nccl/2.12.7-cuda.11.3
module load /data/home/vkhalidov/modulefiles/nccl_efa/1.2.0-nccl.2.12.7-cuda.11.3

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCC_INFO=INFO
export NCCL_SOCKET_IFNAME=ens32


# vit-s with increasing + decreasing schedule to see whether we can accomodate a larger radius without collapse
CACHE="--adv.adv_cache=1 --radius.schedule_type=peak --radius.start_epoch=30 --radius.min_multiplier=0"

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.09411764705 --adv.step_size_input=0.01882352941 --logging.project_name imgnt1K_simplevits_fb --logging.resume_id=fb_simplevits_abd --logging.folder outputs/simplevits/advpyramidcache_r12_peak --dist.world_size=16" &> job_log/run8-5.log &
