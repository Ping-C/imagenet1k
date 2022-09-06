# setting some environment variables for multinode jobs
source /data/home/pingchiang/miniconda/etc/profile.d/conda.sh
conda activate ffcv_v5

module load /data/home/vkhalidov/modulefiles/cuda/11.3
module load /data/home/vkhalidov/modulefiles/nccl/2.12.7-cuda.11.3
module load /data/home/vkhalidov/modulefiles/nccl_efa/1.2.0-nccl.2.12.7-cuda.11.3

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCC_INFO=INFO
export NCCL_SOCKET_IFNAME=ens32

# launch jobs
# Things to review
# 1. Is the world size correct?
# 2. Is the number of workers less than 8?
# 3. check that no job names are the same
# 4. check that no resume_ids are the same
# 5. check that the data path is at the right location

# copy jobs over before starting
mkdir -p /data/home/yipinzhou/imagenet1k/outputs/mvits_imgnt1k
cp -r /data/home/pingchiang/project/ffcv-imagenet/outputs/mvits_imgnt1k/baseline /data/home/yipinzhou/imagenet1k/outputs/mvits_imgnt1k/baseline
cp -r /data/home/pingchiang/project/ffcv-imagenet/outputs/mvits_imgnt1k/pyramid_r0.04_s0.01_c5 /data/home/yipinzhou/imagenet1k/outputs/mvits_imgnt1k/pyramid_r0.04_s0.01_c5
cp -r /data/home/pingchiang/project/ffcv-imagenet/outputs/mvits_imgnt1k/pyramidcache_r0.06_s0.012 /data/home/yipinzhou/imagenet1k/outputs/mvits_imgnt1k/pyramidcache_r0.06_s0.012
cp -r /data/home/pingchiang/project/ffcv-imagenet/outputs/mvits_imgnt1k/pyramidcache_r0.08_s0.016 /data/home/yipinzhou/imagenet1k/outputs/mvits_imgnt1k/pyramidcache_r0.08_s0.016


# mvit jobs resume
# baseline
python -u launch.py --command="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_baseline.yaml --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.folder outputs/mvits_imgnt1k/baseline --logging.resume_id=431ae767-de79-4b04-bf67-4948d4757514 --dist.world_size=16" &> job_log/run3-1.log &
# 5 step
python -u launch.py --command="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_pyramid.yaml --adv.radius_input=0.04 --adv.step_size_input=0.01  --adv.num_steps=5 --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.resume_id=f6b96173-7e71-4788-aa13-53f6da7943cd --logging.folder outputs/mvits_imgnt1k/pyramid_r0.04_s0.01_c5 --dist.world_size=16" &> job_log/run3-2.log &
# adv cach 0.06
python -u launch.py --command="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_pyramid.yaml --adv.radius_input=0.06 --adv.step_size_input=0.012 --adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0 --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.resume_id=mvits_aa --logging.folder outputs/mvits_imgnt1k/pyramidcache_r0.06_s0.012 --dist.world_size=16" &> job_log/run3-3.log &
# adv cach 0.08
python -u launch.py --command="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_pyramid.yaml --adv.radius_input=0.08 --adv.step_size_input=0.016 --adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0 --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.resume_id=68dd0278-b77e-455b-8bdb-55979c0d2533 --logging.folder outputs/mvits_imgnt1k/pyramidcache_r0.08_s0.016 --dist.world_size=16" &> job_log/run3-4.log &


# vitb experiments with large weight decay and new data augmentation to see whether the new baseline is better 
# also run control pyramid adversarial training and cache pyramid adversarial training

CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --adv.radius_input=0.06 --adv.step_size_input=0.012 --logging.project_name imgnt1K_simplevitb_v2 --logging.resume_id=simplevitb1000v2ac --logging.folder outputs/simplevits_1K_v2/advpyramidcache_r0.06 --dist.world_size=16" &> job_log/run3-5.log &
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --adv.radius_input=0.08 --adv.step_size_input=0.016 --logging.project_name imgnt1K_simplevitb_v2 --logging.resume_id=simplevitb1000v2ad --logging.folder outputs/simplevits_1K_v2/advpyramidcache_r0.08 --dist.world_size=16" &> job_log/run3-6.log &

# experiment where only universal adversary is used and no clean examples are shown
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --adv.radius_input=0.02 --adv.step_size_input=0.004 --adv.adv_loss_weight=1 --logging.project_name imgnt1K_simplevitb_v2 --logging.resume_id=simplevitb1000v2af --logging.folder outputs/simplevits_1K_v2/advpyramidcachenoclean_r0.02 --dist.world_size=16" &> job_log/run3-7.log &
