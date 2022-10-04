python visualize_attack_noise.py \
--data_path /data/home/pingchiang/data/imagenet_ffcv/val_500_0.50_90.ffcv \
--model_paths \
outputs/simplevits/baseline/fb_simplevits_aaa/epoch299.pt \
outputs/simplevits/advpyramidcache_r10/fb_simplevits_aag_v2/epoch299.pt \
outputs/simplevits/advpyramid_r6_s3_c2/fb_simplevits_aac/epoch299.pt \
outputs/simplevits/advpyramid_r6_s1_c5/fb_simplevits_aad/epoch299.pt

python visualize_attention_map.py \
--num_images 100 \
--data_path /data/home/pingchiang/data/imagenet_ffcv/val_500_0.50_90.ffcv \
--model_paths \
outputs/simplevits/baseline/fb_simplevits_aaa/epoch299.pt \
outputs/simplevits/advpyramidcache_r10/fb_simplevits_aag_v2/epoch299.pt \
outputs/simplevits/advpyramid_r6_s3_c2/fb_simplevits_aac/epoch299.pt \
outputs/simplevits/advpyramid_r6_s1_c5/fb_simplevits_aad/epoch299.pt