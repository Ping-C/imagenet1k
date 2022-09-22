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

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls.yaml --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=aab --logging.folder outputs/simplevits/baseline --dist.world_size=16" &> job_log/run4-1.log &
