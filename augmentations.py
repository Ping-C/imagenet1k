from kornia.geometry import transform
from abc import ABC, abstractmethod
import torch
class ParameterizedAugmentation(ABC):
 
    @abstractmethod
    def augment(self, image):
        pass
    
    @abstractmethod
    def update(self):
        # update parameters associated the parameterized augmentation using the gradient
        pass


class Warp(ParameterizedAugmentation):
    def __init__(self, radius=0.001, step_size=0.0001, device=None, random=False):
        super().__init__()
        self.parameter_delta = None
        self.step_size = step_size
        self.radius = radius
        self.device = device
        self.random = random

    def augment(self, images):
        if self.parameter_delta is None:
            self.parameter_center = torch.eye(3, dtype=images.dtype).repeat(len(images), 1, 1).to(self.device)
            self.parameter_delta = torch.zeros_like(self.parameter_center)
            self.parameter_delta.requires_grad_()
        
        return transform.warp_perspective(images, self.parameter_center+self.parameter_delta, images.shape[2:])

    def update(self):
        # update parameters associated the parameterized augmentation using the gradient
        if self.random:
            if self.parameter_delta is not None:
                self.parameter_delta.data = torch.randn_like(self.parameter_delta.data).sign() * self.radius
        else:
            if self.parameter_delta is not None and self.parameter_delta.grad is not None:
                self.parameter_delta.data += self.parameter_delta.grad.sign() * self.step_size
                self.parameter_delta.data.clamp_(-self.radius, self.radius)
                self.parameter_delta.grad = None

if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import SimpleRGBImageDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToTorchImage, RandAugment
    from ffcv.writer import DatasetWriter
    from dataclasses import replace
    from typing import Callable, Optional, Tuple
    from ffcv.pipeline.state import State
    from ffcv.transforms.utils.fast_crop import rotate, shear, blend, \
        adjust_contrast, posterize, invert, solarize, equalize, fast_equalize, \
        autocontrast, sharpen, adjust_saturation, translate, adjust_brightness
    import torchvision.transforms as tv
    import matplotlib.pyplot as plt
    import torchvision
    BATCH_SIZE = 64
    IMG_SIZE = 224
    
    loader = Loader('/fsx/pingchiang/imagenet_ffcv_100class/val_400_0.1_90.ffcv', batch_size=BATCH_SIZE,
                    num_workers=2, order=OrderOption.RANDOM,
                    drop_last=True, pipelines={'image': [CenterCropRGBImageDecoder((224, 224), ratio=224/256), ToTensor(), ToTorchImage()]}
                    )

    idx = 0
    for ims, labs in loader:
        break
    
    images = ims.float()/255
    images = images.half()


    aug = Warp()

    for i in range(5):
        import pdb;pdb.set_trace()
        aug_images = aug.augment(images)
        loss = ((aug_images - images)**2).sum()
        print(f"loss{i}", loss)
        loss.backward()
        aug.update()
        aug_images_grid = torchvision.utils.make_grid(aug_images)
        plt.imshow(aug_images_grid.permute(1,2, 0))
        plt.savefig(f"aug{i}.png")
        plt.close()


    






# translate_x, translate_y, shear_x, shear_y, rotate, solarize, equalize, posterize, autocontrast, color, brightness, contrast, sharpness