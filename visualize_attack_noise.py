
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



def get_images(data_path, num_images=20):
    # get data
    IMAGENET_MEAN_ALT = np.array([0.5, 0.5, 0.5]) * 255
    IMAGENET_STD_ALT = np.array([0.5, 0.5, 0.5]) * 255
    DEFAULT_CROP_RATIO = 224/256
    resolution = 224
    this_device = f'cuda:0'
    res_tuple = (resolution, resolution)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    mean = IMAGENET_MEAN_ALT 
    std = IMAGENET_STD_ALT
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(ch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(mean, std, np.float32)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device(this_device),
        non_blocking=True)
    ]

    loader = Loader(data_path,
                    batch_size=num_images,
                    num_workers=2,
                    order=OrderOption.QUASI_RANDOM,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=False)
    images, labels = next(iter(loader))
    return images, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_paths', nargs='+', default=["outputs/simplevits/baseline_v3/aac/epoch299.pt", "outputs/simplevits/baseline_v3/aac/epoch289.pt"],
                        help='models used for visualization')
    parser.add_argument('--data_path', default="/fs/nexus-scratch/pchiang/imagenet_dataset/train_400_0.6_50.ffcv",
                        help='models used for visualization')
    args = parser.parse_args()
    images, labels = get_images(args.data_path)
    num_images= 10
    attack_iters = 10
    step_size = 1/255*2
    radius = 6/255
    fig, axes = plt.subplots(num_images, 2*len(args.model_paths), figsize=(2*4*len(args.model_paths), num_images*4))
    # load model
    for model_i, model_path in enumerate(args.model_paths):
        arch = "vit_s_v8"
        data_path = args.data_path
        resolution = 224
        model = models.get_arch(arch, num_classes=1000)
        model = ch.nn.DataParallel(model).cuda()

        checkpoint_dict = ch.load(model_path)
        model.load_state_dict(checkpoint_dict['state_dict'])       


        adv_noise = ch.zeros_like(images)
        adv_noise.requires_grad = True
        loss_func = ch.nn.CrossEntropyLoss()

        for attack_i in range(attack_iters):
            pred = model(images+adv_noise)
            loss = loss_func(pred, labels)
            loss.backward()
            adv_noise.data += step_size*adv_noise.grad.sign()
            adv_noise.data.clamp_(-radius, +radius)
            adv_noise.grad = None



        for img_i in range(num_images):
            axes[img_i][model_i*2].imshow(images[img_i].permute(1, 2, 0).cpu()/2+0.5)
            adv_noise_cur = adv_noise[img_i].permute(1, 2, 0).detach()
            adv_noise_cur -= adv_noise_cur.min()
            adv_noise_cur /= adv_noise_cur.max()
            adv_noise_cur = adv_noise_cur.cpu()
            axes[img_i][model_i*2+1].imshow(adv_noise_cur )
    os.makedirs('figs', exist_ok = True)
    plt.savefig(f'figs/adv_noise.png')
    plt.close()



