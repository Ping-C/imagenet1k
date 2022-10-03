
import models
import torch as ch
import ffcv
import numpy as np
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os
from visualize_attack_noise import get_images
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_paths', nargs='+', default=["outputs/simplevits/baseline_v3/aac/epoch299.pt", "outputs/simplevits/baseline_v3/aac/epoch289.pt"],
                    help='models used for visualization')
parser.add_argument('--data_path', default="/fs/nexus-scratch/pchiang/imagenet_dataset/train_400_0.6_50.ffcv",
                    help='models used for visualization')
parser.add_argument('--num_images', type=int, default=10,
                    help='number of images to visualize')
args = parser.parse_args()

# get data
images_all, labels = get_images(args.data_path)



num_images= args.num_images
fig, axes = plt.subplots(num_images, 2*len(args.model_paths), figsize=(2*4*len(args.model_paths), num_images*4))
# load model

arch = "vit_s_v8"
for model_i, model_path in enumerate(args.model_paths):
    model = models.get_arch(arch, num_classes=1000)
    # turn on visualization
    for layer_i, layer in enumerate(model.transformer.layers):
        att, ff = layer
        att.vis = True
    
    model = ch.nn.DataParallel(model).cuda()
    checkpoint_dict = ch.load(model_path)
    model.load_state_dict(checkpoint_dict['state_dict'])
    model = model.module

    for img_i in range(num_images):
        images = images_all[img_i:img_i+1]
        pred, att_maps = model(images, get_attention_maps=True)
        
        # loop through each layer
        cum_att = None
        for layer_i in range(len(model.transformer.layers)):
            map = att_maps[:, layer_i] # shape = (batch size, # head, # patches, # of patches)
            b_size, head_cnt, p_cnt, _ = map.shape
            avgmap = map.mean(dim=1) # shape = (batch size,  # of patches, # of patches)
            avgmap = avgmap.cpu()
            residual_att = ch.eye(avgmap.size(1))
            avgmap += residual_att
            avgmap = avgmap/avgmap.sum(dim=2, keepdims=True)
            if cum_att is None:
                cum_att = avgmap
            else:
                cum_att = ch.matmul(avgmap, cum_att)

        axes[img_i][model_i*2].imshow(images[0].permute(1, 2, 0).cpu()/2+0.5)
        axes[img_i][model_i*2+1].imshow(cum_att[0].mean(dim=0).reshape(14, 14).detach())
os.makedirs('figs', exist_ok = True)
plt.savefig(f'figs/attention_map.png')
plt.close()


