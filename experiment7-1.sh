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


python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml        --adv.radius_input=0.04705882352 --adv.step_size_input=0.04705882352 --logging.project_name imgnt1K_simplevitb_fb --logging.resume_id=fb_simplevitb_aab --logging.folder outputs/simplevitb/advpyramid_r6_s6_c1 --adv.num_steps=1 --dist.world_size=16" &> job_log/run7-1.log &

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml        --adv.radius_input=0.04705882352 --adv.step_size_input=0.02352941176 --logging.project_name imgnt1K_simplevitb_fb --logging.resume_id=fb_simplevitb_aac --logging.folder outputs/simplevitb/advpyramid_r6_s3_c2 --adv.num_steps=2 --dist.world_size=16" &> job_log/run7-2.log &

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml        --adv.radius_input=0.04705882352 --adv.step_size_input=0.0156862745 --logging.project_name imgnt1K_simplevitb_fb --logging.resume_id=fb_simplevitb_aad --logging.folder outputs/simplevitb/advpyramid_r6_s1_c5 --adv.num_steps=5 --dist.world_size=16" &> job_log/run7-3.log &

CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.04705882352 --adv.step_size_input=0.0094117647 --logging.project_name imgnt1K_simplevitb_fb --logging.resume_id=fb_simplevitb_aae --logging.folder outputs/simplevitb/advpyramidcache_r6 --dist.world_size=16" &> job_log/run7-4.log &
