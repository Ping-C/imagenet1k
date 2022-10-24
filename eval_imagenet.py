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
import webdataset as wds
import datetime
from randaugment_v2 import RandAugmentV2, RandAugmentV3, RandAugmentV4

class DictChecker(Checker):
    def check(self, value):
        return json.loads(value)

    def help(self):
        return "a dictionary"

class ListChecker(Checker):
    def check(self, value):
        return [v for v in value.split(",")]

    def help(self):
        return "a list"


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
    weight_decay_explicit=Param(int, 'whether to use decoupled weight decay', default=0),
    weight_decay_nolr=Param(int, 'whether to use apply learning rate to weight decay', default=0),
    grad_clip_norm=Param(float, 'gradient clipping threshold', default=None),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    fixed_dropout=Param(int, 'whether to use fixed dropout pattern when running sam', default=0),
    mixup=Param(int, 'mixup augmentation', default=0),
    randaug=Param(int, 'random augmentation', default=0),
    randaug_num_ops=Param(int, 'number of composable random augmentation', default=2),
    randaug_magnitude=Param(int, 'magnitude of random augmentation', default=15),
    randaug_version=Param(str, 'v1, v2, v3', default='v1'),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0),
    freeze_nonlayernorm_epochs=Param(int, 'use blurpool?', default=None),
    mixed_precision=Param(int, 'whether to use mixed precision training', default=1),
    altnorm=Param(int, 'whether to use alternative normalization', default=0),
    fake_bug=Param(int, 'whether to raise a fake exception for debugging purposes', default=0),
)

Section('adv', 'hyper parameter related to adversarial training').params(
    num_steps=Param(int, 'number of adversarial steps'),
    radius_input=Param(float, 'adversarial radius'),
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
    cache_class_wise_targeted=Param(float, 'whether to use class wise universal perturbation', default=0),
    cache_class_wise_targeted_loss=Param(
        str, 
        'different types of targeted loss: negative_ce, reverse_ce, negative_two_logit_ce, reverse_two_logit_ce', default=None),
    cache_class_wise_shuffle_iter=Param(int, 'number of iterations to shuffle class wise adversaries'),
    pyramid=Param(float, 'whether to use pyramid adversary', default=0),
    generator=Param(float, 'whether to use generator adversary', default=0),
    generator_arch=Param(str, 'generator architecture, such as vit_s_generator_flat, vit_s_generator_deep, vit_s_generator_deepflat', default='vit_s_generator'),
    generator_universal=Param(float, 'whether to use generator with randomized noised input', default=0),
    generator_grad_mult=Param(float, 'multiplier for generator gradient', default=1),
    optimizer=Param(str, 'optimizer used to optimizer the image'),
    lr=Param(float, 'optimizer used to optimizer the image'),
    distill_subnet_advloss=Param(float, 'whether to use distill subnet loss', default=0),
    adv_dropout_rate=Param(float, 'dropout rate for adv dropout', default=0),
)
Section('advdropout', 'hyperparameters associated with adversarial dropout').params(
    parameterization=Param(str, 'how the adversarial dropout is parameterized: plain, all, ', default='plain')
)
Section('pyramid', 'hyperparameters associated with pyramid adversarial training').params(
    scale_factors=Param(str, 'scales or pyramid', default='32,16,1'),
    m_factors=Param(str, 'step multiplier', default='20,10,1'),
)

Section('universal_feature_adv', 'hyperparameters associated with universal feature adversary').params(
    step_size=Param(float, 'step size', default=0.001),
    radius=Param(float, 'radius', default=0.01),
    layers=Param(str, 'radius', default='0,'),
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
    multinode=Param(int, 'multinode', default=0),
    rank=Param(int, 'rank of node if not specified grabs the SLURM_PROCID ')
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
    elif schedule_type == 'peak':
        return get_radius_multiplier_linear_peak(epoch)
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
    return 1 + (epoch - start_epoch) / (epochs - start_epoch) * (max_multiplier -1)


@param('radius.min_multiplier')
@param('radius.start_epoch')
@param('training.epochs')
def get_radius_multiplier_linear_decrease(epoch, start_epoch, min_multiplier, epochs):
    if epoch < start_epoch:
        return 1
    return 1 + (epoch - start_epoch) / (epochs - start_epoch) * (min_multiplier -1)

@param('radius.min_multiplier')
@param('radius.start_epoch')
@param('training.epochs')
def get_radius_multiplier_linear_peak(epoch, start_epoch, min_multiplier, epochs):
    # increase the radius from min_multiplier until start_epoch and then decrease the radius until 
    if epoch < start_epoch:
        return (epoch) / start_epoch * (1-min_multiplier) + min_multiplier
    else:
        return 1 + (epoch - start_epoch) / (epochs - start_epoch) * (min_multiplier -1)

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
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.initialize_logger()
        if resume_id is not None or resume_checkpoint is not None:
            # traverse the folder and find the latest checkpoint
            ckpt_path = None
            if resume_checkpoint is None:
                latest_epoch_ckpt_file = None
                latest_epoch = -float('inf')
                if distributed:
                    ch.distributed.barrier()
                for file in os.listdir(self.log_folder):
                    if 'epoch' in file:
                        epoch = int(file.replace('epoch', '').replace('.pt', ''))
                        if epoch > latest_epoch:
                            latest_epoch = epoch
                            latest_epoch_ckpt_file = file
                if latest_epoch != -float('inf'):
                    ckpt_path = Path(self.log_folder)/latest_epoch_ckpt_file
                
            else:
                ckpt_path = resume_checkpoint
            if ckpt_path is not None:
                print(f"loaded checkpoint at :{ckpt_path}")
                checkpoint_dict = ch.load(ckpt_path, map_location='cpu')
                if 'state_dict' in checkpoint_dict:
                    # load model, optimizer, starting epoch number
                    if convert:
                        checkpoint_dict['state_dict'] = self.convert(checkpoint_dict['state_dict'])
                    self.model.load_state_dict(checkpoint_dict['state_dict'], strict=False)
                    self.starting_epoch = checkpoint_dict['starting_epoch']
                else:
                    if convert:
                        checkpoint_dict = self.convert(checkpoint_dict)
                    self.starting_epoch = 0
                    self.model.load_state_dict(checkpoint_dict)
                if 'generator_state_dict' in checkpoint_dict:
                    self.generator.load_state_dict(checkpoint_dict['generator_state_dict'])
            else:
                print(f"Failed to find checkpoint in {self.log_folder}. Starting a new run")
                self.starting_epoch = 0
        else:
            self.starting_epoch = 0
        if rank == 0:
            if resume_id:
                wandb.init(project=project_name, entity="pchiang", name=os.path.normpath(self.log_folder).split("/")[-2], resume=True, id=self.uid,
                config={'.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()})
            else:
                wandb.init(project=project_name, entity="pchiang", name=os.path.normpath(self.log_folder).split("/")[-2], id=self.uid,
                config={'.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()})
            self.usewb = True
        else:
            self.usewb = False
        if adv_augment_on:
            self.prepare_adversarial_augment()
        class_idx = json.load(open('imagenet_class_index.json'))
        self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    
    @param('adv_augment.radius')
    @param('adv_augment.step_size')
    @param('adv_augment.random')
    def prepare_adversarial_augment(self, radius, step_size, random):
        self.adv_augment = Warp(radius=radius, step_size=step_size, device=f'cuda:{self.gpu}', random=random)

    def convert(self, checkpoint_dict):
        convert_dict = dict()
        for k, v in checkpoint_dict.items():
            if "norm" in k or '.net.0.' in k or '.linear_head.0.' in k:
                name_list = k.split('.')
                for i, name in enumerate(name_list):
                    if 'norm' in name or (i-1>=0 and name_list[i-1] == 'net' and name == '0') or  (i-1>=0 and name_list[i-1] == 'linear_head' and name == '0') :
                        break
                name_list.insert(i+1, 'layernorm_clean')
                clean_key = ".".join(name_list)
                name_list[i+1] = 'layernorm_adv'
                adv_key = ".".join(name_list)
                convert_dict[clean_key] = v
                convert_dict[adv_key] = v
            else:
                convert_dict[k] = v
        return convert_dict
    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.rank, world_size=world_size, timeout=datetime.timedelta(seconds=1200))
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.barrier()
        dist.destroy_process_group()
        # put here to explicitly terminate the process
        raise ValueError("a hack to terminate the process")

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'cyclic_warm': get_cyclic_lr_withwarmups,
            'step': get_step_lr,
            'step_fastadvprop': get_step_lr_fastadvprop
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    @param('training.mixup')
    @param('adv.adv_loss_smooth')
    @param('training.weight_decay_explicit')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, mixup, adv_loss_smooth=None, weight_decay_explicit=0):
        if optimizer == 'sgd':

            # Only do weight decay on non-batchnorm parameters
            all_params = list(self.model.named_parameters())
            if hasattr(self, 'generator'):
                all_params += list(self.generator.named_parameters())
            bn_params = [v for k, v in all_params if ('bn' in k)]
            other_params = [v for k, v in all_params if not ('bn' in k)]
            param_groups = [{
                'params': bn_params,
                'weight_decay': 0.
            }, {
                'params': other_params,
                'weight_decay': weight_decay
            }]

            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        elif optimizer == 'adam':
             # Only do weight decay on non-layernorm parameters and non bias parameters
            all_params = list(self.model.named_parameters())
            if hasattr(self, 'generator'):
                all_params += list(self.generator.named_parameters())
            norm_params = []
            bias_params = []
            other_params = []
            for k, v in all_params:
                if 'norm' in k:
                    norm_params.append(v)
                elif 'bias' in k:
                    bias_params.append(v)
                else:
                    other_params.append(v)
            param_groups = [{
                'params': norm_params + bias_params,
                'weight_decay': 0.
            }, {
                'params': other_params,
                'weight_decay': 0 if weight_decay_explicit else weight_decay
            }]

            self.optimizer = ch.optim.Adam(param_groups, lr=1)
        elif optimizer == 'adamw':
            all_params = list(self.model.named_parameters())
            if hasattr(self, 'generator'):
                all_params += list(self.generator.named_parameters())
            norm_params = []
            bias_params = []
            other_params = []
            for k, v in all_params:
                if 'norm' in k:
                    norm_params.append(v)
                    print(f'Parameter {k} is norm')
                elif 'bias' in k:
                    print(f'Parameter {k} is bias')
                    bias_params.append(v)
                else:
                    print(f'Parameter {k} uses weight decay')
                    other_params.append(v)
            param_groups = [{
                'params': norm_params + bias_params,
                'weight_decay': 0.
            }, {
                'params': other_params,
                'weight_decay': 0 if weight_decay_explicit else weight_decay
            }]
            self.optimizer = ch.optim.AdamW(param_groups, lr=1)
        else:
            raise ValueError(f"unsupported optimizer: {optimizer}")
        if mixup:
            self.aux_loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            def mixuploss(output, target):
                loss_1 = self.aux_loss(output, target[:, 0].long())
                loss_2 = self.aux_loss(output, target[:, 1].long())
                lam = target[0, 2]
                loss_train = loss_1 * lam + loss_2 * (1-lam)
                return loss_train
            self.train_loss_func = mixuploss
            self.test_loss_func = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.train_loss_func = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.test_loss_func = self.train_loss_func
        if adv_loss_smooth is None:
            self.train_adv_loss_func = self.train_loss_func
        else:
            def mixupadvloss(output, target):
                loss_1 = ch.nn.CrossEntropyLoss(label_smoothing=adv_loss_smooth)(output, target[:, 0].long())
                loss_2 = ch.nn.CrossEntropyLoss(label_smoothing=adv_loss_smooth)(output, target[:, 1].long())
                lam = target[0, 2]
                loss_train = loss_1 * lam + loss_2 * (1-lam)
                return loss_train
            self.train_adv_loss_func = ch.nn.CrossEntropyLoss(label_smoothing=adv_loss_smooth)
            if mixup:
                self.train_adv_loss_func = mixupadvloss
            else:
                self.train_adv_loss_func = ch.nn.CrossEntropyLoss(label_smoothing=adv_loss_smooth)
        self.attack_adv_loss_func = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def create_train_loader(self):
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
    @param('training.randaug_version')
    @param('training.mixed_precision')
    @param('training.altnorm')
    def create_train_loader_ffcv(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory, mixup, randaug=False, randaug_num_ops=None, randaug_magnitude=None, randaug_version=None, mixed_precision=True,
                            altnorm=False):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        mean = IMAGENET_MEAN_ALT if altnorm else IMAGENET_MEAN
        std = IMAGENET_STD_ALT if altnorm else IMAGENET_STD
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(mean, std, np.float16) if mixed_precision else NormalizeImage(mean, std, np.float32)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]
        if mixup:
            mixup_img = ffcv.transforms.ImageMixup(0.2, True)
            mixup_label = ffcv.transforms.LabelMixup(0.2, True)
            image_pipeline.insert(2, mixup_img)
            label_pipeline.insert(1, mixup_label)
        
        if randaug:
            if randaug_version == 'v1':
                image_pipeline.insert(2, ffcv.transforms.RandAugment(num_ops=randaug_num_ops, magnitude=randaug_magnitude))
            elif randaug_version == 'v2':
                image_pipeline.insert(2, RandAugmentV2(num_ops=randaug_num_ops, magnitude=randaug_magnitude))
            elif randaug_version == 'v3':
                image_pipeline.insert(2, RandAugmentV3(num_ops=randaug_num_ops, magnitude=randaug_magnitude))
            elif randaug_version == 'v4':
                image_pipeline.insert(2, RandAugmentV4(num_ops=randaug_num_ops, magnitude=randaug_magnitude))
            else:
                raise ValueError(f'randaug_version {randaug_version} is not available')
        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader



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
    def create_val_loader_ffcv(self, val_dataset, num_workers, batch_size,
                          resolution, distributed, mixed_precision=1, altnorm=False):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        mean = IMAGENET_MEAN_ALT if altnorm else IMAGENET_MEAN
        std = IMAGENET_STD_ALT if altnorm else IMAGENET_STD
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(mean, std, np.float16) if mixed_precision else NormalizeImage(mean, std, np.float32)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    @param('training.mixed_precision')
    @param('training.altnorm')
    def create_val_loader_torch(self, val_dataset, num_workers, batch_size,
                          resolution, distributed, mixed_precision=1, altnorm=False):
        val_dataset = '/datasets01/imagenet_full_size/061417/val'
        normalize = transforms.Normalize(mean=IMAGENET_MEAN/255,
                                std=IMAGENET_STD/255)

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
    @param('training.fake_bug')
    def train(self, epochs, log_level, save_checkpoint_interval, freeze_nonlayernorm_epochs=None, reset_layernorm=False, fake_bug=0):
        if fake_bug:
            raise ValueError("Random fake bug for testing")
        radius = (6/255*2, 6/255*2, 6/255*2)
        step_sizes = (1/255*2, 3/255*2, 6/255*2)
        steps = (6, 2, 1)
        self.cur_epoch = self.starting_epoch
        for radii, step_size, steps in zip(radius, step_sizes, steps):
            res = self.get_resolution(self.starting_epoch)
            self.decoder.output_size = (res, res)
            print(f"radii: {radii*255/2}, step_size:{step_size*255/2}, steps:{steps}")
            train_stats = self.train_loop(self.starting_epoch, radius_input=radii, step_size_input=step_size, num_steps=steps)

            if log_level > 0:
                extra_dict = train_stats

                self.eval_and_log(extra_dict)
    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        
        self.model.module.define_mask()
        self.model.module.turn_on_masking(mode="adv")
        stats = self.val_loop()
        stats_adv = {f"{k}_adv": v for k, v in stats.items()}
        self.model.module.turn_on_masking(mode="random")
        stats = self.val_loop()
        stats_random = {f"{k}_random": v for k, v in stats.items()}
        self.model.module.turn_off_masking()
        stats = self.val_loop()
        val_time = time.time() - start_val
        
        if self.rank == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],\
                'val_time': val_time
            }, **extra_dict, **stats, **stats_adv, **stats_random))
            if self.usewb:
                if not hasattr(self, 'last_log_time'):
                    self.last_log_time = self.start_time
                wandb.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'val_time': val_time,
                'epoch_time': (time.time() - self.last_log_time)/60,
                'radius_multiplier': get_radius_multiplier(self.cur_epoch)
                }, **extra_dict, **stats, **stats_adv, **stats_random), step=extra_dict['epoch']
                )
                self.last_log_time = time.time()
            

        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    @param('data.num_classes')
    @param('adv.generator')
    @param('adv.generator_arch')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool, num_classes, generator=False, generator_arch=None):
        scaler = GradScaler()
        model = models.get_arch(arch, num_classes=num_classes)
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)
           
        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
        else:
            model = ch.nn.DataParallel(model).cuda()
        
        if generator:
            generator = models.get_arch("vit_s_generator")
            generator = generator.to(memory_format=ch.channels_last)
            generator = generator.to(self.gpu)

            if distributed:
                generator = ch.nn.parallel.DistributedDataParallel(generator, device_ids=[self.gpu])
            else:
                generator = ch.nn.DataParallel(generator).cuda()
            self.generator = generator


        return model, scaler
    

    @param('adv.radius_input')
    @param('adv.step_size_input')
    @param('adv.cache_frequency')
    @param('adv.cache_size_multiplier')
    @param('adv.cache_sequential')
    @param('adv.cache_class_wise')
    @param('adv.cache_class_wise_targeted')
    @param('adv.pyramid')
    @param('data.num_classes')
    @param('training.mixup')
    @param('adv.optimizer')
    @param('adv.lr')
    @param('adv.cache_class_wise_shuffle_iter')
    def adv_cache(self, images, target, step_size_input=None, radius_input=None, 
    cache_frequency=1, cache_size_multiplier=1, cache_class_wise=False, num_classes=None,
    mixup=None, cache_sequential=False, pyramid=0, optimizer='pgd', lr=None, beta=0.9, radius_multiplier=1, cache_class_wise_shuffle_iter=float('inf'), cache_class_wise_targeted=False):
        if pyramid:
            if cache_class_wise_targeted:
                return self.adv_cache_pyramid_classwise_targeted(images, target, radius_multiplier=radius_multiplier)
            else:
                return self.adv_cache_pyramid(images, target, radius_multiplier=radius_multiplier)
            
        if cache_class_wise:
            if not hasattr(self, 'adv_cache_noise'):
                self.adv_cache_noise = ch.zeros((int(num_classes), *images.shape[1:]), device=images.device, dtype=images.dtype)
                self.adv_cache_noise.requires_grad_(True)
                self.adv_cache_iter = 0
            else:
                self.adv_cache_iter += 1 
                
                if self.adv_cache_iter %cache_frequency  == 0:
                    # take an adversarial step
                    if optimizer == 'pgd':
                        grad = self.adv_cache_noise.grad
                        self.adv_cache_noise.data += grad.sign() * step_size_input
                        self.adv_cache_noise.data.clamp_(-radius_input, +radius_input)
                    elif optimizer == 'momentum':
                        grad = self.adv_cache_noise.grad
                        if not hasattr(self, 'adv_cache_noise_last'):
                            self.adv_cache_grad_last = grad
                        else:
                            self.adv_cache_grad_last = self.adv_cache_grad_last * beta + grad * (1-beta)
                        self.adv_cache_noise.data += self.adv_cache_grad_last * lr
                        self.adv_cache_noise.data.clamp_(-radius_input*radius_multiplier, +radius_input*radius_multiplier)
                    self.adv_cache_noise.grad = None
                else:
                    self.adv_cache_noise.grad = None

                if self.adv_cache_iter % cache_class_wise_shuffle_iter == 0:
                    print("shuffling the data set!")
                    self.adv_cache_noise.data = self.adv_cache_noise.data[ch.randperm(len(self.adv_cache_noise.data))]
            if mixup:
                t0 = target[:, 0].long()
                t1 = target[:, 1].long()
                lam = target[0, 2]
                final_noise = self.adv_cache_noise[t0]*lam + self.adv_cache_noise[t1]*(1-lam)
                return final_noise, FeatureNoise({})
            else:
                return self.adv_cache_noise[target], FeatureNoise({})
        else:
            b_size = images.shape[0]
            if not hasattr(self, 'adv_cache_noise'):
                self.adv_cache_noise = ch.zeros((int(b_size*cache_size_multiplier), *images.shape[1:]), device=images.device, dtype=images.dtype)
                self.adv_cache_noise.requires_grad_(True)
                self.adv_cache_iter = 0
            else:
                self.adv_cache_iter += 1
                if self.adv_cache_iter%cache_frequency  == 0:
                    # take an adversarial step
                    if optimizer == 'pgd':
                        grad = self.adv_cache_noise.grad
                        self.adv_cache_noise.data += grad.sign() * step_size_input
                        self.adv_cache_noise.data.clamp_(-radius_input*radius_multiplier, +radius_input*radius_multiplier)
                    elif optimizer == 'momentum':
                        grad = self.adv_cache_noise.grad
                        if not hasattr(self, 'adv_cache_grad_last'):
                            self.adv_cache_grad_last = grad
                        else:
                            self.adv_cache_grad_last = self.adv_cache_grad_last * beta + grad * (1-beta)
                        
                        self.adv_cache_noise.data += self.adv_cache_grad_last * lr
                        self.adv_cache_noise.data.clamp_(-radius_input*radius_multiplier, +radius_input*radius_multiplier)
                    
                    self.adv_cache_noise.grad = None
                else:
                    self.adv_cache_noise.grad = None
            if cache_size_multiplier == 1:
                return self.adv_cache_noise, FeatureNoise({})
            elif cache_size_multiplier > 1:
                if cache_sequential:
                    start_idx = (self.adv_cache_iter%cache_size_multiplier)*b_size
                    end_idx = start_idx + b_size
                    idx = ch.arange(start_idx, end_idx, dtype=ch.long)
                else:
                    idx = ch.randperm(len(self.adv_cache_noise))[:b_size]
                return self.adv_cache_noise[idx], FeatureNoise({})
            elif cache_size_multiplier < 1:
                return self.adv_cache_noise.repeat(int(1/cache_size_multiplier), 1, 1, 1), FeatureNoise({})
    
    @param('adv.radius_input')
    @param('adv.step_size_input')
    @param('adv.cache_frequency')
    @param('adv.cache_size_multiplier')
    @param('adv.cache_sequential')
    @param('adv.cache_class_wise')
    @param('data.num_classes')
    @param('training.mixup')
    def adv_cache_pyramid(self, images, target, step_size_input=None, radius_input=None, 
    cache_frequency=1, cache_size_multiplier=1, cache_class_wise=False, num_classes=None,
    mixup=None, cache_sequential=False, scale_factors=[32, 16, 1], m_factors=[20, 10, 1], radius_multiplier=1):
        b_size = images.shape[0]

        if not hasattr(self, 'adv_cache_noise_pyramid'):
            self.adv_cache_noise_pyramid = []
            for scale_factor in scale_factors:
                if cache_class_wise:
                    noise_cur = ch.zeros((int(num_classes*cache_size_multiplier), images.shape[1], *(ch.tensor(images.shape[2:])//scale_factor)), device=images.device, dtype=images.dtype)
                else:
                    noise_cur = ch.zeros((int(b_size*cache_size_multiplier), images.shape[1], *(ch.tensor(images.shape[2:])//scale_factor)), device=images.device, dtype=images.dtype)
                
                self.adv_cache_noise_pyramid.append(
                    noise_cur
                )
            for adv_cache_noise in self.adv_cache_noise_pyramid:
                adv_cache_noise.requires_grad_(True)
            self.adv_cache_iter = 0
        else:
            self.adv_cache_iter += 1
            if self.adv_cache_iter%cache_frequency  == 0:
                # take an adversarial step
                for adv_cache_noise in self.adv_cache_noise_pyramid:
                    grad = adv_cache_noise.grad
                    adv_cache_noise.data += grad.sign() * step_size_input
                    adv_cache_noise.data.clamp_(-radius_input*radius_multiplier, +radius_input*radius_multiplier)
            for adv_cache_noise in self.adv_cache_noise_pyramid:
                adv_cache_noise.grad = None

        pyramid_noise = None
        for noise, m_factor in zip(self.adv_cache_noise_pyramid,  m_factors):
            if pyramid_noise is None:
                pyramid_noise = ch.nn.functional.upsample(noise, images.shape[2:])*m_factor
            else:
                pyramid_noise += ch.nn.functional.upsample(noise, images.shape[2:])*m_factor
        
        if cache_class_wise:
            if cache_sequential:
                offset = (self.adv_cache_iter%cache_size_multiplier)*num_classes
            else:
                offset = ch.randint(int(cache_size_multiplier), size=(len(images),), device=images.device)*num_classes
            if cache_size_multiplier >= 1:
                if mixup: 
                    t0 = target[:, 0].long()
                    t1 = target[:, 1].long()
                    lam = target[0, 2]
                    final_noise = pyramid_noise[t0+offset]*lam + pyramid_noise[t1+offset]*(1-lam)
                else:
                    final_noise = pyramid_noise[target+offset]
            else:
                raise ValueError("cache_size_multiplier smaller than 1 is not supported")
            
            return final_noise, FeatureNoise({})
        else:
            if cache_size_multiplier == 1:
                return pyramid_noise, FeatureNoise({})
            elif cache_size_multiplier > 1:
                if cache_sequential:
                    start_idx = (self.adv_cache_iter%cache_size_multiplier)*b_size
                    end_idx = start_idx + b_size
                    idx = ch.arange(start_idx, end_idx, dtype=ch.long)
                else:
                    idx = ch.randperm(len(pyramid_noise))[:b_size]
                return pyramid_noise[idx], FeatureNoise({})
            elif cache_size_multiplier < 1:
                return pyramid_noise.repeat(int(1/cache_size_multiplier), 1, 1, 1), FeatureNoise({})

    @param('adv.radius_input')
    @param('adv.step_size_input')
    @param('data.num_classes')
    @param('adv.cache_size_multiplier')
    def adv_cache_pyramid_classwise_targeted(self, images, target, step_size_input=None, radius_input=None, num_classes=None, scale_factors=[32, 16, 1], m_factors=[20, 10, 1], radius_multiplier=1, cache_size_multiplier=1):
        b_size = images.shape[0]

        if not hasattr(self, 'adv_cache_noise_pyramid'):
            self.adv_cache_noise_pyramid = []
            for scale_factor in scale_factors:
                noise = ch.zeros((int(num_classes*cache_size_multiplier), images.shape[1], *(ch.tensor(images.shape[2:])//scale_factor)), device=images.device, dtype=images.dtype)
                noise.requires_grad_(True)
                self.adv_cache_noise_pyramid.append(noise)
            self.adv_cache_iter = 0
        else:
            self.adv_cache_iter += 1
            # take an adversarial step
            for adv_cache_noise in self.adv_cache_noise_pyramid:
                grad = adv_cache_noise.grad
                adv_cache_noise.data += grad.sign() * step_size_input
                adv_cache_noise.data.clamp_(-radius_input*radius_multiplier, +radius_input*radius_multiplier)
            for adv_cache_noise in self.adv_cache_noise_pyramid:
                adv_cache_noise.grad = None
        
        
        pyramid_noise = None
        for noise, m_factor in zip(self.adv_cache_noise_pyramid,  m_factors):
            if pyramid_noise is None:
                pyramid_noise = ch.nn.functional.upsample(noise, images.shape[2:])*m_factor
            else:
                pyramid_noise += ch.nn.functional.upsample(noise, images.shape[2:])*m_factor
        
        target_labels = ch.randint(num_classes, size=(len(images),), device=images.device)
        pyramid_noise = pyramid_noise[target_labels+int(self.adv_cache_iter%cache_size_multiplier)*num_classes]
        return (pyramid_noise, target_labels), FeatureNoise({})
    
    @param('adv.radius_input')
    @param('adv.generator_universal')
    def adv_generator(self, images, target, radius_input, radius_multiplier, generator_universal=False):
        if generator_universal:
            noise = self.generator(ch.randn_like(images))
        else:
            noise = self.generator(images)
        images_adv = (noise-0.5)* 2 * radius_input* radius_multiplier
        return images_adv, FeatureNoise({})
        
    @param('adv.num_steps')
    @param('adv.radius_input')
    @param('adv.step_size_input')
    @param('adv.adv_features')
    @param('adv.adv_cache')
    @param('adv.optimizer')
    @param('adv.lr')
    @param('adv.pyramid')
    @param('training.mixed_precision')
    @param('adv.generator')
    def adv_step(self, model, images, target,
        num_steps=None, step_size_input=None, radius_input=None, adv_features=None, aux_branch=False, adv_cache=False, optimizer='pgd', lr=None, beta=0.9, mixed_precision=True, pyramid=False, radius_multiplier=1, generator=False):
        if generator:
            return self.adv_generator(images, target, radius_multiplier=radius_multiplier)
        if adv_cache:
            return self.adv_cache(images, target, radius_multiplier=radius_multiplier)
        if pyramid:
            return self.adv_step_pyramid(model, images, target, radius_multiplier=radius_multiplier)
        input_adv_noise = ch.zeros_like(images, requires_grad=True)
        feature_adv_noise = FeatureNoise({int(layer): None for layer in adv_features} if adv_features is not None else {})
        for step in range(num_steps):
            with autocast(enabled=bool(mixed_precision)):
                if isinstance(model.module, ResNet) or isinstance(model.module, VisionTransformer):
                    output = model(images+input_adv_noise) # resnet doesn't support feature noise
                else:
                    if aux_branch == False:
                        output = model(images+input_adv_noise, feature_noise=feature_adv_noise)
                    else:
                        output = model(images+input_adv_noise, feature_noise=feature_adv_noise, aux_branch=aux_branch)
                
                loss_adv = self.train_loss_func(output, target)
                all_step_sizes = [step_size_input]
                all_radii = [radius_input]
                all_noises = [input_adv_noise] 
                if adv_features is not None:
                    for k in adv_features:
                        all_noises.append(feature_adv_noise[int(k)])
                        all_step_sizes.append(adv_features[k]['step_size'])
                        all_radii.append(adv_features[k]['radius'])
                all_grad = ch.autograd.grad(self.scaler.get_scale()*loss_adv, all_noises)
                # apply perturbations to each individual features
                if optimizer == 'pgd':
                    for noise, grad, step_size, radius in zip(all_noises, all_grad, all_step_sizes, all_radii):
                        # normalize gradients to unit norm & times the radius
                        # grad /= (grad.norm(dim=ch.arange(1, len(grad.shape)).tolist(), keepdim=True, p=2) + 1e-5)
                        
                        noise.data += grad.sign() * step_size * radius_multiplier
                        noise.data.clamp_(-radius * radius_multiplier, +radius * radius_multiplier)
                elif optimizer == 'momentum':
                    if not hasattr(self, 'last_adv_grad'):
                        self.last_adv_grad = all_grad

                    for noise, grad, last_grad, step_size, radius in zip(all_noises, all_grad, self.last_adv_grad, all_step_sizes, all_radii):
                        # normalize gradients to unit norm & times the radius
                        # grad /= (grad.norm(dim=ch.arange(1, len(grad.shape)).tolist(), keepdim=True, p=2) + 1e-5)
                        
                        noise.data += (last_grad * beta + grad/self.scaler.get_scale() * (1-beta))*lr
                        last_grad.data = (last_grad * beta + grad/self.scaler.get_scale() * (1-beta))
                        noise.data.clamp_(-radius, +radius)

        for k in feature_adv_noise:
            feature_adv_noise[k] = feature_adv_noise[k].detach()
        
        return input_adv_noise.detach(), feature_adv_noise

    @param('adv.num_steps')
    @param('adv.radius_input')
    @param('adv.step_size_input')
    @param('training.mixed_precision')
    @param('pyramid.scale_factors')
    @param('pyramid.m_factors')
    @param('data.num_classes')
    def adv_step_pyramid(self, model, images, target,
        num_steps=None, step_size_input=None, radius_input=None, mixed_precision=True,
        scale_factors=[32, 16, 1], m_factors=[20, 10, 1], num_classes=None, radius_multiplier=1):
        if not isinstance(scale_factors, list):
            scale_factors = [int(v) for v in scale_factors.split(',')]
        if not isinstance(m_factors, list):
            m_factors = [int(v) for v in m_factors.split(',')]
        b_size = images.shape[0]
        all_noises = []
        for scale_factor in scale_factors:
            noise = ch.zeros(
                    (b_size, images.shape[1], *(ch.tensor(images.shape[2:])//scale_factor)), 
                    device=images.device, 
                    dtype=images.dtype)
            noise.requires_grad_(True)
            all_noises.append(
                noise
            )
            
        def compose_pyramid_noise(noise):
            pyramid_noise = None
            for noise, m_factor in zip(noise, m_factors):
                if pyramid_noise is None:
                    pyramid_noise = ch.nn.functional.upsample(noise, images.shape[2:])*m_factor
                else:
                    pyramid_noise += ch.nn.functional.upsample(noise, images.shape[2:])*m_factor
            return pyramid_noise
        rand_target = ch.randint(low=0, high=num_classes, size=(len(images),), device=target.device)
        for step in range(num_steps):
            with autocast(enabled=bool(mixed_precision)):
                pyramid_noise = compose_pyramid_noise(all_noises)
                output = model(images+pyramid_noise)
                loss_adv = self.attack_adv_loss_func(output, rand_target)
                all_grad = ch.autograd.grad(self.scaler.get_scale()*loss_adv, all_noises)
                # apply perturbations to each individual features
                for noise, grad in zip(all_noises, all_grad):
                    noise.data -= grad.sign() * step_size_input * radius_multiplier
                    noise.data.clamp_(-radius_input*radius_multiplier, +radius_input*radius_multiplier)

        return (compose_pyramid_noise(all_noises).detach(), rand_target) , FeatureNoise({})

    @param('logging.log_level')
    @param('training.grad_clip_norm')
    @param('adv.num_steps')
    @param('adv.adv_loss_weight')
    @param('adv.adv_loss_even')
    @param('adv.freeze_layers')
    @param('sam.radius')
    @param('training.fixed_dropout')
    @param('training.mixed_precision')
    @param('adv.split_backward')
    def train_loop(self, epoch, log_level, grad_clip_norm=None, num_steps=0, adv_loss_weight=0, radius=0, fixed_dropout=False, freeze_layers=None, split_backward=False, 
    freeze_nonlayernorm=False, adv_loss_even=False, mixed_precision=True, step_size_input=None, radius_input=None): 
        model = self.model
        model.train()
        losses = []
        losses_adv = []
        losses_clean = []
        train_stats = defaultdict(lambda : 0)
        n_samples = 0
        adv = num_steps > 0
        sam = radius > 0

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = self.train_loader.__len__()
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader, total=iters, disable=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        for ix, (images, target) in enumerate(iterator):
            if ix == 1:
                print("finished first batch...")
            if ix == 3:
                break
            ### Training start

            # generate adversarial examples/intermediate adversarial examples
            if adv:
                if hasattr(self.model.module, 'make_adv'):
                    self.model.module.make_adv()
                images_adv, features_adv = self.adv_step(self.model, images, target, radius_multiplier=get_radius_multiplier(epoch = epoch),
                                                        step_size_input=step_size_input, radius_input=radius_input, num_steps=num_steps)
                
                if len(images_adv) == 2:
                    images_adv, target_adv = images_adv
                else:
                    target_adv = None
            if fixed_dropout:
                fixed_seed = ch.randint(0, 999999, size=(1,), device=images.device)
            if sam:
                with self.model.no_sync():
                    if fixed_dropout:
                        ch.cuda.manual_seed(fixed_seed)
                    output = self.model(images)
                    loss_train = self.train_loss_func(output, target)
                    loss_train.backward()
                    #calculate norm of grad
                    norm = 0
                    for para in self.model.parameters():
                        norm += (para.grad**2).sum()
                    norm = norm**0.5
                    for para in self.model.parameters():
                        para.pert = para.grad/norm*radius
                        para.data += para.pert
                        para.grad = None

            if hasattr(self.model.module, 'make_adv'):
                self.model.module.make_adv()

            # generate adversarial data augmentation if it is a thing
            output_adv = self.model(images+images_adv, feature_noise=features_adv, freeze_layers=freeze_layers)

            loss_train_adv = self.train_adv_loss_func(output_adv, target)
            dummy_loss = sum([para.sum()*0 for para in model.parameters()])


            
            loss_train_adv.backward()
            

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:

            if sam:
                for i, para in enumerate(self.model.parameters()):
                    para.data -= para.pert
                
            ### Training end
            
       
        return train_stats


    @param('validation.lr_tta')
    @param('eval.layernorm_switch')
    @param('training.mixed_precision')
    @param('data.torch_loader')
    def val_loop(self, lr_tta, layernorm_switch, mixed_precision=True, torch_loader=0):
        model = self.model
        model.eval()
        stats = {}
        if hasattr(self.model.module, 'make_clean'):
            self.model.module.make_clean()
        with ch.no_grad():
            with autocast(enabled=bool(mixed_precision)):
                for images, target in tqdm(self.val_loader, disable=True):
                    if torch_loader:
                        images, target = images.cuda(), target.cuda()
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)

                    loss_val = self.test_loss_func(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()} | stats
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.gpu)
        }
        folder = (Path(folder) / str(self.uid)).absolute()
        self.log_folder = folder
        if self.rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }
            params['logging.resume_id'] = str(self.uid)
            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

    @param('training.eval_only')
    def log(self, content, eval_only=False):
        print(f'=> Log: {json.dumps(content, indent=4)}')
        if self.rank != 0: return
        cur_time = time.time()
        if eval_only:
            log_filename = 'log_eval'
        else:
            log_filename = 'log'
        if "adv_distribution" in content:
            del content["adv_distribution"]
        with open(self.log_folder / log_filename, 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    @param('dist.multinode')
    @param('dist.rank')
    def launch_from_args(cls, distributed, world_size, multinode=False, rank=None):
        if distributed:
            if multinode:
                # if using multinode mode, slurm will spawn the needed jobs, but each process needs ot get the rnak from process id
                if rank is None:
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
