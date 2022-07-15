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
import json
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
    save_checkpoint_interval=Param(int, 'intervals for saving checkpoints', default=5)
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
    mixup=Param(int, 'mixup augmentation', default=False),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0)
)
Section('eval', 'evaluation parameters').params(
    checkpoint_path=Param(str, 'path for loading the checkpoints'),
    linear_probe=Param(int, 'whether to evaluate linear probe'),
    gsnr=Param(int, 'whether to eval gsnr')
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
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
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.initialize_logger()
        

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
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
    @param('eval.linear_probe')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, linear_probe):
        if linear_probe:
            if optimizer == 'sgd':

                all_params = list(self.model.named_parameters())            
                probe_params = [v for k, v in all_params if ('probe' in k)]
                self.optimizer = ch.optim.SGD(probe_params, lr=1, momentum=momentum)

            elif optimizer == 'adam':
                # Only do weight decay on non-batchnorm parameters
                all_params = list(self.model.named_parameters())            
                probe_params = [v for k, v in all_params if ('probe' in k)]

                self.optimizer = ch.optim.Adam(probe_params, lr=1)
            elif optimizer == 'adamw':
                all_params = list(self.model.named_parameters())
                
                probe_params = [v for k, v in all_params if ('probe' in k)]
                self.optimizer = ch.optim.AdamW(probe_params, lr=1)
            else:
                raise ValueError(f"unsupported optimizer: {optimizer}")
        else:
            all_params = list(self.model.named_parameters())
            self.optimizer = ch.optim.AdamW(self.model.parameters(), lr=1)
        
        self.train_loss_func = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.test_loss_func = self.train_loss_func
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
    @param('eval.checkpoint_path')
    @param('eval.linear_probe')
    @param('eval.gsnr')
    def train(self, epochs, log_level, checkpoint_path, linear_probe=False, gsnr=False):
        for epoch_ckpt in range(4, 300, 5):
            res = self.get_resolution(epoch_ckpt)
            self.decoder.output_size = (res, res)

            # reset weights
            def reset_weights(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            self.model.apply(reset_weights)

            # reload pretrained checkpoints
            self.model.load_state_dict(
                ch.load(
                f'{checkpoint_path}/epoch{epoch_ckpt}.pt',
            map_location={'cuda:%d' % 0: 'cuda:%d' % self.gpu}), 
            strict=False)

            if linear_probe:
                # train the respective head on frozen features
                for epoch_finetune in range(epochs):
                    self.train_loop(epoch_finetune)
            
            self.model.eval()
            if log_level > 0:
                extra_dict = {
                    'epoch': epoch_ckpt
                }
                if gsnr:
                    extra_dict['gsnr']=self.gsnr()

                self.eval_and_log(extra_dict)

        self.eval_and_log({'epoch':epoch_ckpt})

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'val_time': val_time
            }, **extra_dict, **stats))

        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('data.num_classes')
    @param('eval.linear_probe')
    def create_model_and_scaler(self, arch, pretrained, distributed,  num_classes, linear_probe):
        scaler = GradScaler()
        model = models.get_arch(arch, num_classes=num_classes, probe=linear_probe)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
        else:
            model = ch.nn.DataParallel(model)

        return model, scaler
    

    @param('logging.log_level')
    @param('training.grad_clip_norm')
    def train_loop(self, epoch, log_level, grad_clip_norm):
        model = self.model
        model.train()
        losses = []

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
            loss_dict = {}
            nan_count = {}
            with autocast():
                output, probe_output = self.model(images, get_features=True,  get_linear_probes=True)
                # loss_train = self.train_loss_func(output, target)
                loss_train = (output * 0).abs().sum()
                for k, v in probe_output.items():
                    # filter out any examples with nan output
                    mask = ~v.isnan().any(dim=1)
                    cur_loss =  self.train_loss_func(v[mask], target[mask])
                    loss_train += cur_loss
                    loss_dict[k] = cur_loss
                    nan_count[k] = v.isnan().any(dim=1).sum()


            self.scaler.scale(loss_train).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            for probe in model.module.linear_probes:
                ch.nn.utils.clip_grad_norm_(probe.parameters(), grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
                
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

                    for k in probe_output:
                        names += [f'loss_{k}']
                        values += [f'{loss_dict[k].item():.3f}']
                        names += [f'nan_{k}']
                        values += [f'{nan_count[k].item():.3f}']


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
                if ix == 10:
                    break
            
            for i, para in enumerate(self.model.parameters()):
                dist.all_reduce(para.gradmean, op=ReduceOp.SUM)
                dist.all_reduce(para.gradsquaremean, op=ReduceOp.SUM)
                dist.all_reduce(para.count, op=ReduceOp.SUM)
                para.gradmean /= para.count
                para.gradsquaremean /= para.count
                para.gradvar = para.gradsquaremean - para.gradmean ** 2
                para.gsnr = (para.gradmean**2)/para.gradvar

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
                        feature_bank[key].append(features[key].mean(dim=1))
            labels = ch.cat(labels)
            # calculate distance matrix by layers
            acc_by_layers = [0]*len(feature_bank)
            distance = ch.ones((len(labels), len(labels)))
            for i in tqdm(range(len(feature_bank))):
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
                
                # dist.all_reduce(acc, op=ReduceOp.AVG)       
                print(f"layer:{i} {acc.item()}")
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
        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    output, linear_probes = self.model(images, get_features=True, get_linear_probes=True)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))
                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)
                    for k, v in linear_probes.items():
                        self.val_meters[f'top_1_{k}'](v, target)
                    

                    loss_val = self.test_loss_func(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    @param('model.arch')
    def initialize_logger(self, folder, arch):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.gpu)
        }
        if arch == 'vit_s':
            for i in range(12):
                self.val_meters[f'top_1_{i}'] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)
        elif arch == 'mvit':
            for i in range(10):
                self.val_meters[f'top_1_{i}'] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)

        if self.gpu == 0:
            folder = (Path(folder) / str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

    def log(self, content):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
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
    def launch_from_args(cls, distributed, world_size):
        if distributed:
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
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
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
