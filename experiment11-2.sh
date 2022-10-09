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

python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_aeb --logging.folder outputs/simplevits/advpyramidcachetargeted_r6_advw0.01_reversece --adv.cache_class_wise_targeted=1 --adv.adv_loss_weight=0.01 --adv.radius_input=0.04705882352 --adv.step_size_input=0.0094117647 --adv.cache_class_wise_targeted_loss=reverse_ce --adv.pyramid=1 --dist.world_size=16" &> job_log/run11-2.log &


python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_aec --logging.folder outputs/simplevits/advpyramidcachetargeted_r6_advw0.05_reversece --adv.cache_class_wise_targeted=1 --adv.adv_loss_weight=0.05 --adv.radius_input=0.04705882352 --adv.step_size_input=0.0094117647 --adv.cache_class_wise_targeted_loss=reverse_ce --adv.pyramid=1 --dist.world_size=16" &> job_log/run11-3.log &


python -u launch.py --command="python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_aed --logging.folder outputs/simplevits/advpyramidcachetargeted_r6_advw0.1_reversece --adv.cache_class_wise_targeted=1 --adv.adv_loss_weight=0.1 --adv.radius_input=0.04705882352 --adv.step_size_input=0.0094117647 --adv.cache_class_wise_targeted_loss=reverse_ce --adv.pyramid=1 --dist.world_size=16" &> job_log/run11-4.log &
