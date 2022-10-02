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

# vitb with stronger (2, 15) random augmentation
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_randaug2,15.yaml --logging.project_name imgnt1K_simplevitb_fb --logging.resume_id=fb_simplevitb_aal --logging.folder outputs/simplevitb/baseline_(2,15) --dist.world_size=16" &> job_log/run8-1.log &

# vit-s with increasing + decreasing schedule to see whether we can accomodate a larger radius without collapse
CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=peak --radius.start_epoch=30 --radius.min_multiplier=0"

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.06274509803 --adv.step_size_input=0.0125490196 --logging.project_name imgnt1K_simplevits_fb --logging.resume_id=fb_simplevits_abb    --logging.folder outputs/simplevits/advpyramidcache_r8_peak --dist.world_size=16" &> job_log/run8-3.log &

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.07843137254 --adv.step_size_input=0.0156862745 --logging.project_name imgnt1K_simplevits_fb --logging.resume_id=fb_simplevits_abc --logging.folder outputs/simplevits/advpyramidcache_r10_peak --dist.world_size=16" &> job_log/run8-4.log &

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.09411764705 --adv.step_size_input=0.01882352941 --logging.project_name imgnt1K_simplevits_fb --logging.resume_id=fb_simplevits_abd --logging.folder outputs/simplevits/advpyramidcache_r12_peak --dist.world_size=16" &> job_log/run8-5.log &

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.0156862745 --adv.step_size_input=0.0031372549 --adv.adv_loss_weight=1 --logging.project_name imgnt1K_simplevits_fb --logging.resume_id=fb_simplevits_abe --logging.folder outputs/simplevits/advpyramidcachenoclean_r2_peak --dist.world_size=16" &> job_log/run8-6.log &

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.03137254901 --adv.step_size_input=0.0062745098 --adv.adv_loss_weight=1 --logging.project_name imgnt1K_simplevits_fb --logging.resume_id=fb_simplevits_abf --logging.folder outputs/simplevits/advpyramidcachenoclean_r4_peak --dist.world_size=16" &> job_log/run8-7.log &

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.06274509803 --adv.step_size_input=0.0125490196 --adv.adv_loss_weight=1 --logging.project_name imgnt1K_simplevits_fb --logging.resume_id=fb_simplevits_abg --logging.folder outputs/simplevits/advpyramidcachenoclean_r8_peak --dist.world_size=16" &> job_log/run8-8.log &

# vit-s increase the training length of universal adversarial training by 2 and 3 times to see whether we can increase performance
CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=peak --radius.start_epoch=60 --radius.min_multiplier=0 --training.epochs=600"
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.07843137254 --adv.step_size_input=0.0156862745 --logging.project_name imgnt1K_simplevits_fb --logging.resume_id=fb_simplevits_abh --logging.folder outputs/simplevits/advpyramidcache_r10_e600 --dist.world_size=16" &> job_log/run8-9.log &
CACHE="--adv.adv_cache=1 --adv.radius_schedule=1 --radius.schedule_type=peak --radius.start_epoch=120 --radius.min_multiplier=0 --training.epochs=1200"
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE        --adv.radius_input=0.07843137254 --adv.step_size_input=0.0156862745 --logging.project_name imgnt1K_simplevits_fb --logging.resume_id=fb_simplevits_abi --logging.folder outputs/simplevits/advpyramidcache_r10_e600 --dist.world_size=16" &> job_log/run8-10.log &


# regularized pyramid with decreasing radius
SCHEDULE="--adv.radius_schedule=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0"
python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml     $SCHEDULE   --adv.radius_input=0.04705882352 --adv.step_size_input=0.0156862745 --logging.project_name imgnt1K_simplevits_fb --logging.resume_id=fb_simplevits_bba --logging.folder outputs/simplevits/advpyramid_r6_s1_c5_decrease --adv.num_steps=5 --dist.world_size=16" &> job_log/run8-11.log &
