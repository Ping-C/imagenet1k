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
# 3. 
# mvit multi step job
python -u launch.py --command="python train_imagenet.py --config-file configs/mvit_decoupled/mvits_1000cls_advinput_pyramid.yaml --adv.radius_input=0.06 --adv.step_size_input=0.012 --adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0 --training.altnorm=1 --logging.project_name mvits_imgnt1K --logging.resume_id=mvits_aa --logging.folder outputs/mvits_imgnt1k/pyramidcache_r0.06_s0.012 --dist.world_size=16" &> job_log/run2-1.log &

# vitb experiments with large weight decay and new data augmentation to see whether the new baseline is better 
# also run control pyramid adversarial training and cache pyramid adversarial training

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls.yaml                                                                      --logging.project_name imgnt1K_simplevitb_v2 --logging.resume_id=simplevitb1000v2aa --logging.folder outputs/simplevits_1K_v2/baseline --dist.world_size=16" &> job_log/run2-1.log &
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml        --adv.radius_input=0.04 --adv.step_size_input=0.04 --logging.project_name imgnt1K_simplevitb_v2 --logging.resume_id=simplevitb1000v2ab --logging.folder outputs/simplevits_1K_v2/advpyramid_r0.04_s0.04_c1 --dist.world_size=16" &> job_log/run2-1.log &

CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --adv.radius_input=0.06 --adv.step_size_input=0.012 --logging.project_name imgnt1K_simplevitb_v2 --logging.resume_id=simplevitb1000v2ac --logging.folder outputs/simplevits_1K_v2/advpyramidcache_r0.06 --dist.world_size=16" &> job_log/run2-1.log &
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --adv.radius_input=0.08 --adv.step_size_input=0.016 --logging.project_name imgnt1K_simplevitb_v2 --logging.resume_id=simplevitb1000v2ad --logging.folder outputs/simplevits_1K_v2/advpyramidcache_r0.08 --dist.world_size=16" &> job_log/run2-1.log &

# experiment where only universal adversary is used and no clean examples are shown
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --adv.radius_input=0.08 --adv.step_size_input=0.016 --adv.adv_loss_weight=1 --logging.project_name imgnt1K_simplevitb_v2 --logging.resume_id=simplevitb1000v2ad --logging.folder outputs/simplevits_1K_v2/advpyramidcache_r0.08 --dist.world_size=16" &> job_log/run2-1.log &
