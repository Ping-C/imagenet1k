from multiprocessing.sharedctypes import Value
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ReduceOp
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
import models
from collections import defaultdict
import ffcv
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from fastargs.validation import Checker
from models import SimpleViTDecoupledLN, MViTDecoupled, SimpleViTTripleLN
import json
import os
from random import randrange
import wandb
from torchvision.models import ResNet
from torchvision.utils import make_grid
from augmentations import Warp

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

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
from timm.models.vision_transformer import VisionTransformer
Section('model', 'model details').params(
    arch=Param(str, default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_classes=Param(int, 'number of classes in dataset', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
    torch_loader=Param(int, 'whether to use torch loader', default=0)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(str, default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
    warmup_epochs=Param(int, 'number of warmup steps', default=None),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1), 
    save_checkpoint_interval=Param(int, 'intervals for saving checkpoints', default=5), 
    resume_id=Param(str, 'resume id', default=None), 
    resume_checkpoint=Param(str, 'resume path for checkpoint', default=None),
    convert=Param(int, 'whether to convert the model from regular model to decoupled model', default=0),
    usewb=Param(int, 'whether to use weight and bias', default=1),
    project_name=Param(str, 'project name for w&b', default="cache-advprop")
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd', 'adam', 'adamw'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    grad_clip_norm=Param(float, 'gradient clipping threshold', default=None),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    fixed_dropout=Param(int, 'whether to use fixed dropout pattern when running sam', default=0),
    mixup=Param(int, 'mixup augmentation', default=0),
    randaug=Param(int, 'random augmentation', default=0),
    randaug_num_ops=Param(int, 'number of composable random augmentation', default=2),
    randaug_magnitude=Param(int, 'magnitude of random augmentation', default=15),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0),
    freeze_nonlayernorm_epochs=Param(int, 'use blurpool?', default=None),
    mixed_precision=Param(int, 'whether to use mixed precision training', default=1),
    altnorm=Param(int, 'whether to use alternative normalization', default=0),
)

Section('adv', 'hyper parameter related to adversarial training').params(
    num_steps=Param(int, 'number of adversarial steps'),
    radius_input=Param(float, 'adversarial radius'),
    radius_schedule=Param(int, 'whether to vary the radius according to a schedule', default=0),
    step_size_input=Param(float, 'step size for adversarial step'),
    adv_features=Param(DictChecker(), 'attacked feature layers'),
    adv_loss_weight=Param(float, 'weight assigned to adversarial loss'),
    adv_loss_even=Param(float, 'whether to assign the same adversarial loss to '),
    adv_loss_smooth=Param(float, 'weight assigned to adversarial loss'),
    freeze_layers=Param(int, 'number of layers to freeze when conducting adversarial training', default=None),
    split_backward=Param(int, 'splitting two backward pass', default=0),
    adv_cache=Param(int, 'whether to use cache adv strategy', default=0),
    cache_frequency=Param(int, 'whether to use cache adv strategy', default=1),
    cache_size_multiplier=Param(float, 'how large is the cached noise relative to the batch size', default=1),
    cache_sequential=Param(float, 'whether to update the cache sequentially or randomly when cache_size_multiplier is larger than 1', default=0),
    cache_class_wise=Param(float, 'whether to use class wise universal perturbation', default=0),
    cache_class_wise_shuffle_iter=Param(int, 'number of iterations to shuffle class wise adversaries'),
    pyramid=Param(float, 'whether to use class wise universal perturbation', default=0),
    optimizer=Param(str, 'optimizer used to optimizer the image'),
    lr=Param(float, 'optimizer used to optimizer the image'),
)

Section('adv_augment', 'hyper parameter related to adversarial augmentation training').params(
    adv_augment_on=Param(int, 'turn on adversarial augment', default=0),
    radius=Param(float, 'adversarial radius'),
    step_size=Param(float, 'adversarial step size'),
    random=Param(int, 'random augmentation as opposed to adversarial', default=0),
)
Section('radius', 'hyperparameters related to radius scheduling').params(
    schedule_type=Param(str, 'linear_increase, wave, linear decrease'),
    start_epoch=Param(int, 'epochs where one start to implement the radius schedule'),
    period_count=Param(int, 'number of periods within the wave schedule'),
    min_multiplier=Param(float, 'minimum radius during schedules'),
    max_multiplier=Param(float, 'maximum radius multiplier during schedule')
)
Section('sam', 'hyper parameter related to sam training').params(
    radius=Param(float, 'adversarial radius'),
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355'),
    multinode=Param(int, 'multinode', default=0)
)


Section('eval', 'special eval flags').params(
    layernorm_switch=Param(int, 'number gpus', default=0)
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

IMAGENET_MEAN_ALT = np.array([0.5, 0.5, 0.5]) * 255
IMAGENET_STD_ALT = np.array([0.5, 0.5, 0.5]) * 255
DEFAULT_CROP_RATIO = 224/256

# from collections.abc import MutableMapping
class FeatureNoise(object):
    def __init__(self, noise):
        self.noise = noise
    def __getitem__(self, key):
        return self.noise[key]
    def __setitem__(self, key, value):
        self.noise[key] = value
    def __delitem__(self, key):
        del self.noise[key]
    def __iter__(self):
        return iter(self.noise)
    def __len__(self):
        return len(self.noise)
    def __contains__(self, key):
        return key in self.noise
@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = ch.randperm(batch_size).cuda()
    else:
        index = ch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    lam = ch.ones_like(y_a) * lam
    target = ch.cat((y_a[:, None], y_b[:, None], lam[:, None]), dim=1)
    return mixed_x, target

@param('radius.schedule_type')
def get_radius_multiplier(epoch, schedule_type=None):
    if schedule_type is None:
        return 1
    if schedule_type == 'linear_increase':
        return get_radius_multiplier_linear_increase(epoch)
    elif schedule_type == 'linear_decrease':
        return get_radius_multiplier_linear_decrease(epoch)
    elif schedule_type == 'wave':
        return get_radius_multiplier_wave(epoch)
    elif schedule_type == 'teeth':
        return get_radius_multiplier_teeth(epoch)

@param('radius.max_multiplier')
@param('radius.start_epoch')
@param('training.epochs')
def get_radius_multiplier_linear_increase(epoch, start_epoch, max_multiplier, epochs):
    if epoch < start_epoch:
        return 1
    return 1 + (epoch - start_epoch) / epochs * (max_multiplier -1)


@param('radius.min_multiplier')
@param('radius.start_epoch')
@param('training.epochs')
def get_radius_multiplier_linear_decrease(epoch, start_epoch, min_multiplier, epochs):
    if epoch < start_epoch:
        return 1
    return 1 + (epoch - start_epoch) / epochs * (min_multiplier -1)

@param('radius.start_epoch')
@param('training.epochs')
@param('radius.max_multiplier')
@param('radius.min_multiplier')
@param('radius.period_count')
def get_radius_multiplier_wave(epoch, start_epoch, max_multiplier, min_multiplier, epochs, period_count):
    if epoch < start_epoch:
        return 1   
    epoch_per_period = (epochs-start_epoch)//period_count
    epoch_within_a_period = (epoch - start_epoch) % epoch_per_period
    phase_length = epoch_per_period/4
    if epoch_within_a_period <= phase_length:
        return 1+(max_multiplier - 1)*epoch_within_a_period/phase_length
    elif epoch_within_a_period <= phase_length*3:
        return max_multiplier - (max_multiplier - min_multiplier)*(epoch_within_a_period-phase_length)/(phase_length*2)
    elif epoch_within_a_period <= phase_length*4:
        return min_multiplier + (1 - min_multiplier)*(epoch_within_a_period-phase_length*3)/phase_length

@param('radius.start_epoch')
@param('training.epochs')
@param('radius.max_multiplier')
@param('radius.min_multiplier')
@param('radius.period_count')
def get_radius_multiplier_teeth(epoch, start_epoch, max_multiplier, min_multiplier, epochs, period_count):
    if epoch < start_epoch:
        return 1

    epoch_per_period = (epochs-start_epoch)//period_count
    epoch_within_period = (epoch-start_epoch)%epoch_per_period
    if epoch < epoch_per_period + start_epoch:
        return 1+(max_multiplier-1)*epoch_within_period/epoch_per_period
    else:
        return min_multiplier+(max_multiplier-min_multiplier)*epoch_within_period/epoch_per_period

@param('radius.start_epoch')
@param('training.epochs')
@param('radius.period_count')
def reset_layernorm_check(epoch, start_epoch,  epochs, period_count):

    epoch_per_period = (epochs-start_epoch)//period_count
    epoch_within_period = (epoch-start_epoch)%epoch_per_period
    if epoch_within_period == 0 and epoch >= start_epoch + epoch_per_period:
        return True
    else:
        return False



@param('lr.lr')
@param('lr.step_ratio')
@param('training.epochs')
def get_step_lr_fastadvprop(epoch, lr, step_ratio, epochs):
    if epoch >= epochs:
        return 0
    if epoch < 30:
        num_steps = 0
    elif epoch < 60:
        num_steps = 1
    elif epoch < 90:
        num_steps = 2
    elif epoch < 100:
        num_steps = 3
    else:
        num_steps = 4
    
    return step_ratio**num_steps * lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


@param('lr.lr')
@param('training.epochs')
@param('lr.warmup_epochs')
def get_cyclic_lr_withwarmups(epoch, lr, warmup_epochs, epochs):
    xs = [0, warmup_epochs]
    xs.extend([epoch for epoch in range(warmup_epochs+1, epochs)])
    ys = [1e-4 * lr, lr]
    for e in range(warmup_epochs+1, epochs):
        progress = (e-warmup_epochs)/(epochs-warmup_epochs)
        ys.append(lr * 0.5 * (1. + np.cos(np.pi * progress)))

    return np.interp([epoch], xs, ys)[0]

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

class ImageNetTrainer:
    @param('training.distributed')
    @param('logging.resume_id')
    @param('logging.convert')
    @param('logging.resume_checkpoint')
    @param('logging.usewb')
    @param('logging.project_name')
    @param('adv_augment.adv_augment_on')
    def __init__(self, rank, distributed, resume_id = None, convert=False, resume_checkpoint=None, usewb=False, project_name=None,
    adv_augment_on=False):
        self.all_params = get_current_config()
        self.rank = rank
        self.gpu = self.rank % ch.cuda.device_count()

        print("rank:", self.rank, ",gpu:", self.gpu, ',device count:', ch.cuda.device_count())
        if resume_id is None:
            self.uid = str(uuid4())
        else:
            self.uid = resume_id

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
    
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



    @param('data.torch_loader')
    def create_train_loader(self, torch_loader=0):
        if torch_loader:
            return self.create_train_loader_torch()
        else:
            return self.create_train_loader_ffcv()

    
    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    @param('training.mixup')
    @param('training.randaug')
    @param('training.randaug_num_ops')
    @param('training.randaug_magnitude')
    @param('training.mixed_precision')
    @param('training.altnorm')
    def create_train_loader_torch(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory, mixup, randaug=False, randaug_num_ops=None, randaug_magnitude=None, mixed_precision=True,
                            altnorm=False):


        normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                     std=IMAGENET_STD)

        train_path = Path(train_dataset)
        transforms_list = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        if randaug:
            transforms_list.insert(2, transforms.RandAugment(num_ops=randaug_num_ops, magnitude=randaug_magnitude))
        train_dataset = datasets.ImageFolder(
            train_dataset,
            transforms.Compose(transforms_list))
        train_sampler = ch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=8, rank=self.gpu)
        train_loader = ch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=False, sampler=train_sampler)
        # still missing mixup
        return train_loader


    @param('data.torch_loader')
    def create_val_loader(self, torch_loader=0):
        if torch_loader:
            return self.create_val_loader_torch()
        else:
            return self.create_val_loader_ffcv()


    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    @param('training.mixed_precision')
    @param('training.altnorm')
    def create_val_loader_torch(self, val_dataset, num_workers, batch_size,
                          resolution, distributed, mixed_precision=1, altnorm=False):

        normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                std=IMAGENET_STD)

        val_path = Path(val_dataset)
        transforms_list = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        val_dataset = datasets.ImageFolder(
            val_path,
            transforms.Compose(transforms_list))
        val_sampler = ch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        val_loader = ch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=(val_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=val_sampler)
        
        return val_loader

    @param('training.epochs')
    @param('logging.log_level')
    @param('logging.save_checkpoint_interval')
    @param('training.freeze_nonlayernorm_epochs')
    def train(self, epochs, log_level, save_checkpoint_interval, freeze_nonlayernorm_epochs=None, reset_layernorm=False):
        for epoch in range(0, epochs):
            iterator = tqdm(self.train_loader)
            for ix, (images, target) in enumerate(iterator):
                images, target = images.cuda(), target.cuda()



    @param('validation.lr_tta')
    @param('eval.layernorm_switch')
    @param('training.mixed_precision')
    def val_loop(self, lr_tta, layernorm_switch, mixed_precision=True):
        model = self.model
        model.eval()
        stats = {}
        if isinstance(self.model.module, SimpleViTDecoupledLN) or isinstance(self.model.module, MViTDecoupled) :
            self.model.module.make_clean()
        with ch.no_grad():
            for images, target in tqdm(self.val_loader):
                images, target = images.cuda(), target.cuda()
        return stats

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
    @param('training.eval_only')
    def exec(cls, rank, distributed, eval_only):
        trainer = cls(rank=rank)
        if eval_only:
            trainer.eval_and_log()
        else:
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
