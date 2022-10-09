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


CACHE="--adv.adv_cache=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0.1"

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_test_acb --logging.folder outputs/simplevits/advpyramidcachetargeted_r8 --adv.cache_class_wise_targeted=1 --adv.adv_loss_even=1 --adv.radius_input=0.06274509803 --adv.step_size_input=0.0125490196  --adv.pyramid=1 --dist.world_size=16" &> job_log/run9-2.log &


python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_test_acc --logging.folder outputs/simplevits/advpyramidcachetargeted_r10 --adv.cache_class_wise_targeted=1 --adv.adv_loss_even=1 --adv.radius_input=0.07843137254 --adv.step_size_input=0.0156862745  --adv.pyramid=1 --dist.world_size=16" &> job_log/run9-3.log &


CACHE="--adv.adv_cache=1 --radius.schedule_type=peak --radius.start_epoch=30 --radius.min_multiplier=0.1"

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_test_acd --logging.folder outputs/simplevits/advpyramidcachetargeted_r6_peak --adv.cache_class_wise_targeted=1 --adv.adv_loss_even=1 --adv.radius_input=0.04705882352 --adv.step_size_input=0.0094117647  --adv.pyramid=1 --dist.world_size=16" &> job_log/run9-4.log &


python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_test_ace --logging.folder outputs/simplevits/advpyramidcachetargeted_r8_peak --adv.cache_class_wise_targeted=1 --adv.adv_loss_even=1 --adv.radius_input=0.06274509803 --adv.step_size_input=0.0125490196  --adv.pyramid=1 --dist.world_size=16" &> job_log/run9-5.log &


python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vitb_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_test_acf --logging.folder outputs/simplevits/advpyramidcachetargeted_r10_peak --adv.cache_class_wise_targeted=1 --adv.adv_loss_even=1 --adv.radius_input=0.07843137254 --adv.step_size_input=0.0156862745  --adv.pyramid=1 --dist.world_size=16" &> job_log/run9-6.log &
