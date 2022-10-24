
module load /data/home/vkhalidov/modulefiles/cuda/11.3
module load /data/home/vkhalidov/modulefiles/nccl/2.12.7-cuda.11.3
module load /data/home/vkhalidov/modulefiles/nccl_efa/1.2.0-nccl.2.12.7-cuda.11.3

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCC_INFO=INFO
export NCCL_SOCKET_IFNAME=ens32

fn(){
    echo $1*2 | bc -l
}




# python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --logging.project_name test8gpus --logging.resume_id=test_8gpus --logging.folder outputs/simplevits/advpyramid_r6_s2_c3_advw0.5_8gpus --adv.num_steps=3 --adv.adv_loss_weight=0.5 --adv.radius_input=$(fn '6/255') --adv.step_size_input=$(fn '6/255/3')  --adv.pyramid=1 --dist.world_size=8 --training.distributed=0

CACHE="--adv.adv_cache=1 --radius.schedule_type=linear_decrease --radius.start_epoch=30 --radius.min_multiplier=0.1"

# python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_agi_3 --logging.folder outputs/simplevits/advpyramidcachetargeted_r8_advw0.01_neg2ce --adv.cache_class_wise_targeted=1 --adv.adv_loss_weight=0.01 --adv.radius_input=$(fn '8/255') --adv.step_size_input=$(fn '8/255/5') --adv.cache_class_wise_targeted_loss=negative_twologits_ce --adv.pyramid=1 --dist.world_size=8 --training.batch_size=128


# python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_agi_4 --logging.folder outputs/simplevits/advpyramidcachetargeted_r32_advw0.01_neg2ce --adv.cache_class_wise_targeted=1 --adv.adv_loss_weight=0.001 --adv.radius_input=$(fn '32/255') --adv.step_size_input=$(fn '32/255/10') --adv.cache_class_wise_targeted_loss=negative_twologits_ce --adv.pyramid=1 --dist.world_size=8 --training.batch_size=128


# python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb_test --logging.resume_id=fb_simplevits_agj_test4 --logging.folder outputs/simplevits/advpyramidgeneratoruniversal_r8 --adv.adv_loss_weight=0.5 --adv.radius_input=$(fn '8/255') --adv.step_size_input=$(fn '8/255') --adv.pyramid=1 --dist.world_size=8 --training.batch_size=128 --adv.generator=1 --adv.generator_universal=1 --data.train_dataset=/data/home/pingchiang/data/imagenet_ffcv/val_500_0.50_90.ffcv

# python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb_test --logging.resume_id=fb_simplevits_agj_test5 --logging.folder outputs/simplevits/advpyramidadvdrop_r0 --adv.adv_loss_weight=0.5 --adv.radius_input=$(fn '0.001/255') --adv.step_size_input=$(fn '0.001/255') --adv.pyramid=1 --dist.world_size=8 --training.batch_size=128 --data.train_dataset=/data/home/pingchiang/data/imagenet_ffcv/val_500_0.50_90.ffcv --model.arch=vit_s_advdropout --adv.distill_subnet_advloss=1


# python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_ahh --logging.folder outputs/simplevits/advpyramidcache_advdropout_r8 --adv.adv_loss_weight=0.5 --adv.radius_input=$(fn '8/255') --adv.step_size_input=$(fn '8/255') --adv.pyramid=1 --dist.world_size=8 --training.batch_size=128 --model.arch=vit_s_advdropout --adv.distill_subnet_advloss=1


# python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_ahi --logging.folder outputs/simplevits/advpyramidcache_advdropout0.1_r0 --adv.adv_loss_weight=0.5 --adv.radius_input=$(fn '0/255') --adv.step_size_input=$(fn '0/255') --adv.pyramid=1 --dist.world_size=8 --training.batch_size=128 --model.arch=vit_s_advdropout --adv.distill_subnet_advloss=1 --adv.adv_dropout_rate=0.1


# python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_ahj --logging.folder outputs/simplevits/advpyramidcache_advdropout0.1_r0_advw0.1 --adv.adv_loss_weight=0.1 --adv.radius_input=$(fn '0/255') --adv.step_size_input=$(fn '0/255') --adv.pyramid=1 --dist.world_size=8 --training.batch_size=128 --model.arch=vit_s_advdropout --adv.distill_subnet_advloss=1 --adv.adv_dropout_rate=0.1


# python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml $CACHE --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_ahk --logging.folder outputs/simplevits/advpyramidcache_advdropout0.1_r0_advw0.1_v2 --adv.adv_loss_weight=0.1 --adv.radius_input=$(fn '0/255') --adv.step_size_input=$(fn '0/255') --adv.pyramid=1 --dist.world_size=8 --training.batch_size=128 --model.arch=vit_s_advdropout --adv.distill_subnet_advloss=1 --adv.adv_dropout_rate=0.1 --advdropout.parameterization=plain

# python train_imagenet.py --config-file configs/simplevit/vits_1000cls_advpyramid.yaml --logging.project_name imgnet1K_simplevits_fb --logging.resume_id=fb_simplevits_ahk_test --logging.folder outputs/simplevits/test --adv.adv_loss_weight=0.1 --adv.radius_input=$(fn '0/255') --adv.step_size_input=$(fn '0/255') --adv.pyramid=1 --dist.world_size=2 --training.batch_size=32 --model.arch=vit_s_advdropout --adv.distill_subnet_advloss=1 --adv.adv_dropout_rate=0.1 --advdropout.parameterization=lv1 --training.distributed=0

python eval_imagenet.py --logging.folder outputs/simplevits/advpyramid_r6_s1_c5_test --logging.resume_id=test --logging.usewb=0 --data.train_dataset=/data/home/pingchiang/data/imagenet_ffcv/train_500_0.50_90.ffcv --data.val_dataset=/data/home/pingchiang/data/imagenet_ffcv/val_500_0.50_90.ffcv --data.num_classes=1000 --data.num_workers=4 --data.in_memory=1 --training.distributed=0 --model.arch=vit_s_subnet --advdropout.parameterization=lv1 --logging.resume_checkpoint=outputs/simplevits/advpyramid_r6_s1_c5/fb_simplevits_aad/epoch104.pt --training.epochs=300 --adv.num_steps=5 --resolution.max_res=224 --resolution.min_res=224 --adv.step_size=$(fn '1/255') --adv.radius_input=$(fn '6/255') --adv.pyramid=1 --training.mixed_precision=0 --training.batch_size=32 --adv.adv_dropout_rate=0.1

