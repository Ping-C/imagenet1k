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
from models import SimpleViTDecoupledLN, MViTDecoupled
import json
import os
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
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic', 'cyclic_warm']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
    warmup_epochs=Param(int, 'number of warmup steps', default=None),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1), 
    save_checkpoint_interval=Param(int, 'intervals for saving checkpoints', default=5), 
    resume_id=Param(str, 'resume id', default=None)
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
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0)
)

Section('adv', 'hyper parameter related to adversarial training').params(
    num_steps=Param(int, 'number of adversarial steps'),
    radius_input=Param(float, 'adversarial radius'),
    step_size_input=Param(float, 'step size for adversarial step'),
    adv_features=Param(DictChecker(), 'attacked feature layers'),
    adv_loss_weight=Param(float, 'weight assigned to adversarial loss'),
    adv_loss_smooth=Param(float, 'weight assigned to adversarial loss'),
    freeze_layers=Param(int, 'number of layers to freeze when conducting adversarial training', default=None),
    split_layer=Param(int, 'the index of layer after which the model is split into two', default=None),
    flip=Param(int, 'the index of layer after which the model is split into two', default=0),
    split_backward=Param(int, 'splitting two backward pass', default=0)
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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
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
    def __init__(self, rank, distributed, resume_id = None):
        self.all_params = get_current_config()
        self.rank = rank
        self.gpu = self.rank % ch.cuda.device_count()
        print("rank:", self.rank, "gpu:", self.gpu, 'DEVICECOUNT', ch.cuda.device_count())
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
        if resume_id is not None:
            # traverse the folder and find the latest checkpoint
            latest_epoch_ckpt_file = None
            latest_epoch = -float('inf')
            for file in os.listdir(self.log_folder):
                if 'epoch' in file:
                    epoch = int(file.replace('epoch', '').replace('.pt', ''))
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        latest_epoch_ckpt_file = file
            print(f"latest_epoch_path:{Path(self.log_folder)/latest_epoch_ckpt_file}")
            checkpoint_dict = ch.load(Path(self.log_folder)/latest_epoch_ckpt_file)

            # load model, optimizer, starting epoch number
            self.model.load_state_dict(checkpoint_dict['state_dict'])
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
            self.starting_epoch = checkpoint_dict['starting_epoch']
        else:
            self.starting_epoch = 0
        

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

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'cyclic_warm': get_cyclic_lr_withwarmups,
            'step': get_step_lr
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
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, mixup, adv_loss_smooth=None):
        if optimizer == 'sgd':

            # Only do weight decay on non-batchnorm parameters
            all_params = list(self.model.named_parameters())
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
             # Only do weight decay on non-batchnorm parameters
            all_params = list(self.model.named_parameters())
            bn_params = [v for k, v in all_params if ('bn' in k)]
            other_params = [v for k, v in all_params if not ('bn' in k)]
            param_groups = [{
                'params': bn_params,
                'weight_decay': 0.
            }, {
                'params': other_params,
                'weight_decay': weight_decay
            }]

            self.optimizer = ch.optim.Adam(param_groups, lr=1)
        elif optimizer == 'adamw':
            all_params = list(self.model.named_parameters())
            bn_params = [v for k, v in all_params if ('bn' in k)]
            other_params = [v for k, v in all_params if not ('bn' in k)]
            param_groups = [{
                'params': bn_params,
                'weight_decay': 0.
            }, {
                'params': other_params,
                'weight_decay': weight_decay
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
            self.train_adv_loss_func = ch.nn.CrossEntropyLoss(label_smoothing=adv_loss_smooth)
    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    @param('training.mixup')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory, mixup):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
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

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
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

    @param('training.epochs')
    @param('logging.log_level')
    @param('logging.save_checkpoint_interval')
    def train(self, epochs, log_level, save_checkpoint_interval):
        for epoch in range(self.starting_epoch, epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            train_loss = self.train_loop(epoch)

            if log_level > 0:
                if epoch % 10 == 0:
                    extra_dict = {
                        'train_loss': train_loss,
                        'epoch': epoch
                    }
                else:
                    extra_dict = {
                        'train_loss': train_loss,
                        'epoch': epoch
                    }

                self.eval_and_log(extra_dict)
            if self.rank == 0 and (epoch+1) % save_checkpoint_interval == 0:
                checkpoint_dict = {
                    'state_dict': self.model.state_dict(),
                    'starting_epoch': epoch+1,
                    'optimizer': self.optimizer.state_dict(),
                    'args': {'.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()}
                }
                ch.save(checkpoint_dict, self.log_folder / f'epoch{epoch}.pt')
                
        self.eval_and_log({'epoch':epoch})
        if self.rank == 0:
            checkpoint_dict = {
                    'state_dict': self.model.state_dict(),
                    'starting_epoch': epoch+1,
                    'optimizer': self.optimizer.state_dict(),
                    'args': {'.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()}
                }
            ch.save(checkpoint_dict, self.log_folder / f'final_weights.pt')

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.rank == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_time': val_time
            }, **extra_dict))

        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    @param('data.num_classes')
    @param('adv.split_layer')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool, num_classes, split_layer=None):
        scaler = GradScaler()
        model = models.get_arch(arch, num_classes=num_classes, split_layer=split_layer)
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
        return model, scaler
    

    @param('adv.num_steps')
    @param('adv.radius_input')
    @param('adv.step_size_input')
    @param('adv.adv_features')
    def adv_step(self, model, images, target,
        num_steps=None, step_size_input=None, radius_input=None, adv_features=None, aux_branch=False):
        input_adv_noise = ch.zeros_like(images, requires_grad=True)
        feature_adv_noise = FeatureNoise({int(layer): None for layer in adv_features} if adv_features is not None else {})
        for step in range(num_steps):
            with autocast():
                if aux_branch == False:
                    output = self.model(images+input_adv_noise, feature_noise=feature_adv_noise)
                else:
                    output = self.model(images+input_adv_noise, feature_noise=feature_adv_noise, aux_branch=aux_branch)
                loss_adv = self.train_loss_func(output, target)
                all_step_sizes = [step_size_input]
                all_radii = [radius_input]
                all_noises = [input_adv_noise] 
                if adv_features is not None:
                    for k in adv_features:
                        all_noises.append(feature_adv_noise[int(k)])
                        all_step_sizes.append(adv_features[k]['step_size'])
                        all_radii.append(adv_features[k]['radius'])
                all_grad = ch.autograd.grad(loss_adv, all_noises)
                # apply perturbations to each individual features
                for noise, grad, step_size, radius in zip(all_noises, all_grad, all_step_sizes, all_radii):
                    # normalize gradients to unit norm & times the radius
                    # grad /= (grad.norm(dim=ch.arange(1, len(grad.shape)).tolist(), keepdim=True, p=2) + 1e-5)
                    noise.data += grad.sign() * step_size
                    noise.data.clamp_(-radius, +radius)
        for k in feature_adv_noise:
            feature_adv_noise[k] = feature_adv_noise[k].detach()
        
        return input_adv_noise.detach(), feature_adv_noise

    @param('logging.log_level')
    @param('training.grad_clip_norm')
    @param('adv.num_steps')
    @param('adv.adv_loss_weight')
    @param('adv.freeze_layers')
    @param('sam.radius')
    @param('training.fixed_dropout')
    @param('adv.split_layer')
    @param('adv.flip')
    @param('adv.split_backward')
    def train_loop(self, epoch, log_level, grad_clip_norm, num_steps=0, adv_loss_weight=0, radius=0, fixed_dropout=False, freeze_layers=None, split_layer=None, flip=False, split_backward=False):
        model = self.model
        model.train()
        losses = []
        adv = num_steps > 0
        sam = radius > 0

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]
            self.optimizer.zero_grad(set_to_none=True)

            # generate adversarial examples/intermediate adversarial examples
            if adv:
                if flip:
                    images_adv, features_adv = self.adv_step(self.model, images, target, aux_branch=True)
                else:
                    if isinstance(self.model.module, SimpleViTDecoupledLN) or isinstance(self.model.module, MViTDecoupled) :
                        self.model.module.make_adv()
                    images_adv, features_adv = self.adv_step(self.model, images, target)
            if fixed_dropout:
                fixed_seed = ch.randint(0, 999999, size=(1,), device=images.device)
            with autocast():
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
                
                if flip:
                    output = self.model(images, aux_branch=True)
                    loss_train_aux = self.train_loss_func(output, target)
                    output = self.model(images+images_adv, feature_noise=features_adv, aux_branch=True)
                    loss_train_aux_adv = self.train_loss_func(output, target)
                    loss_train_aux = loss_train_aux_adv * adv_loss_weight + loss_train_aux * (1-adv_loss_weight)
                    
                    output = self.model(images, freeze_layers=freeze_layers)            
                    loss_train = self.train_loss_func(output, target)
                    loss_train += loss_train_aux
                else:
                    if split_layer:
                        # use a different loss for clean examples when using two head model
                        # clean loss through the auxilary branch
                        output = self.model(images, aux_branch=True)
                        loss_train_aux = self.train_loss_func(output, target)
                        
                        # clean loss through the main branch with freezing
                        output_freeze = self.model(images, freeze_layers=freeze_layers)
                        loss_train = self.train_loss_func(output_freeze, target)
                    else:
                        if isinstance(self.model.module, SimpleViTDecoupledLN) or isinstance(self.model.module, MViTDecoupled) :
                            self.model.module.make_clean()
                        output = self.model(images)
                        loss_train = self.train_loss_func(output, target)
                        if split_backward:
                            # add dummy loss from parameters
                            dummy_loss = 0
                            for para in model.parameters():
                                dummy_loss += para.sum()*0
                            self.scaler.scale(loss_train*(1-adv_loss_weight)+dummy_loss).backward()
                    if adv:
                        if isinstance(self.model.module, SimpleViTDecoupledLN) or isinstance(self.model.module, MViTDecoupled) :
                            self.model.module.make_adv()
                        output_adv = self.model(images+images_adv, feature_noise=features_adv, freeze_layers=freeze_layers)
                        loss_train_adv = self.train_adv_loss_func(output_adv, target)
                        if split_backward:
                            dummy_loss = 0
                            for para in model.parameters():
                                dummy_loss += para.sum()*0
                            self.scaler.scale(loss_train_adv*(adv_loss_weight)+dummy_loss).backward()
                        else:
                            loss_train = loss_train_adv * adv_loss_weight + loss_train * (1-adv_loss_weight)
                    if split_layer:
                        loss_train += loss_train_aux
            if not split_backward:
                self.scaler.scale(loss_train).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            ch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if sam:
                for i, para in enumerate(self.model.parameters()):
                    para.data -= para.pert
                
            ### Training end

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)
            ### Logging end
        return (sum(losses)/len(losses)).item()

    def gsnr(self):
        # return gsnr of each layers
        # loop through a certain number of training examples
        # backward with no sync
        # calculate mean & square mean of gradient of each parameters
        # aggregate the gsnr by layer
        iterator = tqdm(self.train_loader)
        with autocast():
            for ix, (images, target) in enumerate(iterator):
                for i in range(len(target)):
                    image_cur = images[i:i+1]
                    target_cur = target[i:i+1]
                    with self.model.no_sync():
                        output = self.model(image_cur)
                        loss_train = self.train_loss_func(output, target_cur)
                        loss_train.backward()
                        for para in self.model.parameters():
                            if hasattr(para, 'gradmean'):
                                para.gradmean += para.grad
                            else:
                                para.gradmean = para.grad
                            if hasattr(para, 'gradsquaremean'):
                                para.gradsquaremean += para.grad ** 2
                            else:
                                para.gradsquaremean = para.grad ** 2
                            if hasattr(para, 'count'):
                                para.count += 1
                            else:
                                para.count = ch.tensor([1], device=images.device)
                            para.grad = None
                if ix == 100:
                    break
            for i, para in enumerate(self.model.parameters()):
                dist.all_reduce(para.gradmean, op=ReduceOp.SUM)
                dist.all_reduce(para.gradsquaremean, op=ReduceOp.SUM)
                dist.all_reduce(para.count, op=ReduceOp.SUM)
                para.gradmean /= para.count
                para.gradsquaremean /= para.count
                para.gradvar = para.gradsquaremean - para.gradmean ** 2
                para.gsnr = (para.gradmean**2)/para.gradvar
                del para.gradmean
                del para.gradsquaremean
                del para.count
                delattr(para, 'gradmean')
                delattr(para, 'gradsquaremean')
                delattr(para, 'gradvar')
            
            gsnrs = []
            for li, layer in enumerate(self.model.module.transformer.layers):
                gsnr = None
                count = None
                for para in layer.parameters():
                    if gsnr is None:
                        gsnr = para.gsnr.sum()
                        count = para.gsnr.numel()
                    else:
                        gsnr += para.gsnr.sum()
                        count += para.gsnr.numel()
                    del para.gsnr
                    delattr(para, 'gsnr')
                gsnr /= count
                gsnrs.append(gsnr.detach().item())
            return gsnrs

    def knn(self):
        # loop through certain number of training examples
        # extract per layer features
        # calculate the distance matrix
        # calculate knn accuracy by layers
        feature_bank = defaultdict(list)
        labels = []
        with ch.no_grad():
            iterator = tqdm(self.train_loader)
            with autocast():
                for ix, (images, target) in enumerate(iterator):
                    output, features = self.model(images, get_features=True)
                    labels.append(target)
                    for key in features:
                        batch_size = len(output)
                        feature_bank[key].append(features[key].view(batch_size, -1))
                    if ix == 50:
                        break
            labels = ch.cat(labels)
            # calculate distance matrix by layers
            acc_by_layers = [0]*len(feature_bank)
            distance = ch.ones((len(labels), len(labels)), device=images.device)
            for i in range(len(feature_bank)):
                distance[:] = - float('inf') 
                vs = feature_bank[i]
                b_size = len(vs[0])
                for v in vs:
                    v /= v.norm(dim=1, keepdim=True, p=2)
                for v1_i, v1 in enumerate(vs):
                    for v2_i, v2 in enumerate(vs):
                        distance[v1_i*b_size:v1_i*b_size+b_size, v2_i*b_size:v2_i*b_size+b_size] = v1 @ v2.T
                distance[ch.arange(len(distance)), ch.arange(len(distance))] = -float('inf')
                knn = distance.topk(20, largest=True)
                predictions = labels[knn.indices].mode(dim=1).values
                acc = (predictions==labels).float().mean()
                
                dist.all_reduce(acc, op=ReduceOp.AVG)                
                acc_by_layers[i] = acc.item()
            # clean up
            del distance
            for i in range(len(feature_bank)):
                for v in feature_bank[i]:
                    del v
                del feature_bank[i]
            del feature_bank
        return acc_by_layers

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()
        if isinstance(self.model.module, SimpleViTDecoupledLN) or isinstance(self.model.module, MViTDecoupled) :
            self.model.module.make_clean()
        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)

                    loss_val = self.test_loss_func(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
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

    def log(self, content):
        print(f'=> Log: {content}')
        if self.rank != 0: return
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
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
