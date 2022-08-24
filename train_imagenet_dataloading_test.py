from multiprocessing.sharedctypes import Value
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ReduceOp

from models.simple_vit_decoupled import TransformerDecoupledLN
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser
# import models
from collections import defaultdict
from torchvision.transforms import functional as F

import ffcv
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from fastargs.validation import Checker
# from models import SimpleViTDecoupledLN, MViTDecoupled, SimpleViTTripleLN
import json
import os
from random import randrange
from torchvision.models import ResNet
from torchvision.utils import make_grid

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import webdataset as wds
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '/data/home/pingchiang/project/big_vision')
# import big_vision
import big_vision.input_pipeline as input_pipeline
# import big_vision.pp.builder as pp_builder
from big_vision.pp.ops_image import get_decode_jpeg_and_inception_crop, get_random_flip_lr, get_randaug
from big_vision.pp.ops_general_torch import get_onehot, get_keep, get_value_range
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
def _preprocess_fn(data):
    """The preprocessing function that is returned."""

    # Apply all the individual steps in sequence.
    ops = [
        get_decode_jpeg_and_inception_crop(size=224),
        get_random_flip_lr(),
        get_randaug(2, 10),
        get_value_range(-1, 1),
        get_onehot(1000,
               key="label",
               key_result="labels"),
        get_keep("image", "labels")
    ]
    for op in ops:
      print(op)
      data = op(data)

    return data

class DictChecker(Checker):
    def check(self, value):
        return json.loads(value)

    def help(self):
        return "a dictionary"


from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from models.mvit_decoupled import LayerNormDecoupled


Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training'),
    num_workers=Param(int, 'The number of workers')
)


Section('training', 'training hyper param stuff').params(
    batch_size=Param(int, 'The batch size', default=512),
    epochs=Param(int, 'number of epochs', default=30),
    distributed=Param(int, 'distributed training', default=1)
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355'),
    multinode=Param(int, 'multinode', default=0)
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

IMAGENET_MEAN_ALT = np.array([0.5, 0.5, 0.5]) * 255
IMAGENET_STD_ALT = np.array([0.5, 0.5, 0.5]) * 255
DEFAULT_CROP_RATIO = 224/256



class ImageNetTrainer:
    @param('training.distributed')
    def __init__(self, rank, distributed):
        self.all_params = get_current_config()
        self.rank = rank
        self.gpu = self.rank % ch.cuda.device_count()

        print("rank:", self.rank, ",gpu:", self.gpu, ',device count:', ch.cuda.device_count())

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
    
    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.rank, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()


    
    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    def create_train_loader(self, train_dataset=None, num_workers=None, batch_size=None,
                            distributed=None):
        
        train_ds = input_pipeline.make_for_train_torch(
            dataset='imagenet2012_folder',
            split='train',
            batch_size=128,
            preprocess_fn=_preprocess_fn,
            shuffle_buffer_size=250_000,
            cache_raw=True,
            data_dir="/data/home/pingchiang/project/big_vision/imagenet2012")


        
        return train_ds


    @param('training.epochs')
    def train(self, epochs):
        iterator = tqdm(self.train_loader)
        print(f"process : {self.gpu} start looping through epoch")
        try:
            print(f"world size: {ch.distributed.get_world_size()}")
        except:
            pass
        total_size = 0
        for epoch in range(epochs):
            for ix, batch in enumerate(iterator):
                if True:
                    images = ch.from_numpy(np.asarray(batch['image']))
                    images = images.permute(0, 3, 1, 2)
                    target = ch.from_numpy(np.asarray(batch['labels']))
                images, target = images.cuda(), target.cuda()
                total_size += len(images)
                if ix == 0:
                    clean_img_arr = make_grid(images, normalize=True).cpu()
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(10,10))
                    plt.imshow(F.to_pil_image(clean_img_arr))
                    plt.savefig(f'test_{self.rank}_{epoch}.png')
                if ix == 100:
                    break
            print(f"epoch: {epoch }, process : {self.rank} saw {total_size} images target sum{target.sum()}")



    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    @param('dist.multinode')
    def launch_from_args(cls, distributed, world_size, multinode=False):
        if distributed:
            if multinode:
                # if using multinode mode, slurm will spawn the needed jobs, but each process needs ot get the rnak from process id
                rank = os.environ['SLURM_PROCID']
                print(f"executing program for rank {rank}")
                cls.exec(rank=int(rank))
            else:
                ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    def exec(cls, rank, distributed):
        trainer = cls(rank=rank)
        
        trainer.train()

        if distributed:
            trainer.cleanup_distributed()

# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    ImageNetTrainer.launch_from_args()
    