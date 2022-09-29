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


CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.06274509803 --adv.step_size_input=0.0125490196 --logging.project_name imgnt1K_simplevitb_fb --logging.resume_id=fb_simplevitb_aaf    --logging.folder outputs/simplevitb/advpyramidcache_r8 --dist.world_size=16" &> job_log/run7-5.log &

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.07843137254 --adv.step_size_input=0.0156862745 --logging.project_name imgnt1K_simplevitb_fb --logging.resume_id=fb_simplevitb_aag --logging.folder outputs/simplevitb/advpyramidcache_r10 --dist.world_size=16" &> job_log/run7-6.log &

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.0156862745
 --adv.step_size_input=0.0031372549 --adv.adv_loss_weight=1 --logging.project_name imgnt1K_simplevitb_fb --logging.resume_id=fb_simplevitb_aah --logging.folder outputs/simplevitb/advpyramidcachenoclean_r2 --dist.world_size=16" &> job_log/run7-7.log &
 
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.03137254901
 --adv.step_size_input=0.0062745098 --adv.adv_loss_weight=1 --logging.project_name imgnt1K_simplevitb_fb --logging.resume_id=fb_simplevitb_aai --logging.folder outputs/simplevitb/advpyramidcachenoclean_r4 --dist.world_size=16" &> job_log/run7-8.log &

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.06274509803
 --adv.step_size_input=0.0125490196 --adv.adv_loss_weight=1 --logging.project_name imgnt1K_simplevitb_fb --logging.resume_id=fb_simplevitb_aaj --logging.folder outputs/simplevitb/advpyramidcachenoclean_r8 --dist.world_size=16" &> job_log/run7-9.log &
 
 