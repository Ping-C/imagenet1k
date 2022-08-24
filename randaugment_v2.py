import numpy as np
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation, AllocationQuery
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.state import State
from ffcv.transforms.utils.fast_crop import rotate, shear, blend, \
    adjust_contrast, posterize, invert, solarize, equalize, fast_equalize, \
    autocontrast, sharpen, adjust_saturation, translate, adjust_brightness

import ctypes
from numba import njit, prange
import numpy as np
from ffcv.libffcv import ctypes_resize, ctypes_rotate, ctypes_shear, \
    ctypes_add_weighted, ctypes_equalize, ctypes_unsharp_mask

class RandAugmentV2(Operation):
    def __init__(self, 
                 num_ops: int = 2, 
                 magnitude: int = 9, 
                 num_magnitude_bins: int = 31):
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        num_bins = num_magnitude_bins
        # index, name (for readability); bins, sign multiplier
        # those with a -1 can have negative magnitude with probability 0.5
        self.op_table = [
            (0, "Identity", np.array(0.0), 1),
            (1, "ShearX", np.linspace(0.0, 0.3, num_bins), -1),
            (2, "ShearY", np.linspace(0.0, 0.3, num_bins), -1),
            (3, "TranslateX", np.linspace(0.0, 150.0 / 331.0, num_bins), -1),
            (4, "TranslateY", np.linspace(0.0, 150.0 / 331.0, num_bins), -1),
            (5, "Rotate", np.linspace(0.0, 30.0, num_bins), -1),
            (6, "Brightness", np.linspace(0.0, 0.9, num_bins), -1),
            (7, "Color", np.linspace(0.0, 0.9, num_bins), -1),
            (8, "Contrast", np.linspace(0.0, 0.9, num_bins), -1),
            (9, "Sharpness", np.linspace(0.0, 0.9, num_bins), -1),
            (10, "Posterize", 8 - (np.arange(num_bins) / ((num_bins - 1) / 4 / 3)).round(), 1),
            (11, "Solarize", np.linspace(255.0, 0.0, num_bins), 1),
            (12, "AutoContrast", np.array(0.0), 1),
            (13, "Equalize", np.array(0.0), 1),
            (14, "CutOut", np.linspace(0, 100, num_bins), 1),
        ]

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        op_table = self.op_table
        magnitudes = np.array([(op[2][self.magnitude] if op[2].ndim > 0 else 0) for op in self.op_table])
        is_signed = np.array([op[3] for op in self.op_table])
        num_ops = self.num_ops
        def randaug(im, mem):
            dst, scratch, lut, scratchf = mem
            for i in my_range(im.shape[0]):
                for n in range(num_ops):
                    if n == 0:
                        src = im
                    else:
                        src = dst
                        
                    idx = np.random.randint(low=0, high=14+1)
                    mag = magnitudes[idx]
                    if np.random.random() < 0.5:
                        mag = mag * is_signed[idx] 

                    # Not worth fighting numba at the moment.
                    # TODO
                    if idx == 0:
                        dst[i][:] = src[i]
                    
                    if idx == 1: # ShearX (0.004)
                        shear(src[i], dst[i], mag, 0)

                    if idx == 2: # ShearY
                        shear(src[i], dst[i], 0, mag)

                    if idx == 3: # TranslateX
                        translate(src[i], dst[i], int(src[i].shape[1] * mag), 0)

                    if idx == 4: # TranslateY
                        translate(src[i], dst[i], 0, int(src[i].shape[2] * mag))

                    if idx == 5: # Rotate
                        rotate(src[i], dst[i], mag)

                    if idx == 6: # Brightness
                        adjust_brightness(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 7: # Color
                        adjust_saturation(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 8: # Contrast
                        adjust_contrast(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 9: # Sharpness
                        sharpen(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 10: # Posterize
                        posterize(src[i], int(mag), dst[i])

                    if idx == 11: # Solarize
                        solarize(src[i], scratch[i][0], mag, dst[i])

                    if idx == 12: # AutoContrast (TODO: takes 0.04s -> 0.052s) (+0.01s)
                        autocontrast(src[i], scratchf[i][0], dst[i])
                    
                    if idx == 13: # Equalize (TODO: +0.008s)
                        equalize(src[i], lut[i], dst[i])
                    if idx == 14: # cutout
                        cutout(src[i], dst[i], int(mag))
                
            return dst

        randaug.is_parallel = True
        return randaug

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        h, w, c = previous_state.shape
        return replace(previous_state, shape=previous_state.shape), [
            AllocationQuery(previous_state.shape, dtype=np.dtype('uint8')), 
            AllocationQuery((1, h, w, c), dtype=np.dtype('uint8')),
            AllocationQuery((c, 256), dtype=np.dtype('int16')),
            AllocationQuery((1, h, w, c), dtype=np.dtype('float32')),
        ]


class RandAugmentV3(Operation):
    def __init__(self, 
                 num_ops: int = 2, 
                 magnitude: int = 9, 
                 num_magnitude_bins: int = 31):
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        num_bins = num_magnitude_bins
        # index, name (for readability); bins, sign multiplier
        # those with a -1 can have negative magnitude with probability 0.5
        self.op_table = [
            (1, "ShearX", np.linspace(0.1, 1.8*3+0.1, num_bins), -1),
            (2, "ShearY", np.linspace(0.1, 1.8*3+0.1, num_bins), -1),
            (3, "TranslateX", np.linspace(0.0, 300, num_bins), -1),
            (4, "TranslateY", np.linspace(0.0, 300, num_bins), -1),
            (5, "Rotate", np.linspace(0.0, 90.0, num_bins), -1),
            (6, "Brightness", np.linspace(0.1, 1.8*3+0.1, num_bins), 1),
            (7, "Color", np.linspace(0.1, 1.8*3+0.1, num_bins), 1),
            (8, "Contrast", np.linspace(0.1, 1.8*3+0.1, num_bins), 1),
            (9, "Sharpness", np.linspace(0.1, 1.8*3+0.1, num_bins), 1),
            (10, "Posterize", 8 - (np.arange(num_bins) / ((num_bins - 1) / 4)).round(), 1),
            (11, "Solarize", np.linspace(0, 255*3, num_bins), 1),
            (12, "AutoContrast", np.array(0.0), 1),
            (13, "Equalize", np.array(0.0), 1),
            (14, "CutOut", np.linspace(0, 120, num_bins), 1),
            (15, "Invert",  np.array(0.0), 1),
            (16, "SolarizeAdd", np.linspace(0, 110*3, num_bins), 1),
        ]

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        op_table = self.op_table
        magnitudes = np.array([(op[2][self.magnitude] if op[2].ndim > 0 else 0) for op in self.op_table])
        is_signed = np.array([op[3] for op in self.op_table])
        num_ops = self.num_ops
        def randaug(im, mem):
            dst, scratch, lut, scratchf = mem
            for i in my_range(im.shape[0]):
                for n in range(num_ops):
                    if n == 0:
                        src = im
                    else:
                        src = dst
                        
                    idx = np.random.randint(low=1, high=15+1)
                    mag = magnitudes[idx]
                    if np.random.random() < 0.5:
                        mag = mag * is_signed[idx] 

                    # Not worth fighting numba at the moment.
                    # TODO
                    if idx == 0:
                        dst[i][:] = src[i]
                    
                    if idx == 1: # ShearX (0.004)
                        shear(src[i], dst[i], mag, 0)

                    if idx == 2: # ShearY
                        shear(src[i], dst[i], 0, mag)

                    if idx == 3: # TranslateX
                        translate(src[i], dst[i], int(mag), 0)

                    if idx == 4: # TranslateY
                        translate(src[i], dst[i], 0, int(mag))

                    if idx == 5: # Rotate
                        rotate(src[i], dst[i], mag)

                    if idx == 6: # Brightness
                        adjust_brightness(src[i], scratch[i][0], mag, dst[i])

                    if idx == 7: # Color
                        adjust_saturation(src[i], scratch[i][0], mag, dst[i])

                    if idx == 8: # Contrast
                        adjust_contrast(src[i], scratch[i][0], mag, dst[i])

                    if idx == 9: # Sharpness
                        sharpen(src[i], scratch[i][0], mag, dst[i])

                    if idx == 10: # Posterize
                        posterize(src[i], int(mag), dst[i])

                    if idx == 11: # Solarize
                        solarize(src[i], scratch[i][0], mag, dst[i])

                    if idx == 12: # AutoContrast (TODO: takes 0.04s -> 0.052s) (+0.01s)
                        autocontrast(src[i], scratchf[i][0], dst[i])
                    
                    if idx == 13: # Equalize (TODO: +0.008s)
                        equalize(src[i], lut[i], dst[i])
                    
                    if idx == 14: # cutout
                        cutout(src[i], dst[i], int(mag))

                    if idx == 15: # invert
                        invert(src[i], dst[i])
                    
                    if idx == 16: # Solarize
                        solarize_add(src[i], scratch[i][0], mag, dst[i])

                
            return dst

        randaug.is_parallel = True
        return randaug

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        h, w, c = previous_state.shape
        return replace(previous_state, shape=previous_state.shape), [
            AllocationQuery(previous_state.shape, dtype=np.dtype('uint8')), 
            AllocationQuery((1, h, w, c), dtype=np.dtype('uint8')),
            AllocationQuery((c, 256), dtype=np.dtype('int16')),
            AllocationQuery((1, h, w, c), dtype=np.dtype('float32')),
        ]



@njit(parallel=False, fastmath=True, inline='always')
def cutout(source, destination, crop_size):
    coord = (
        np.random.randint(source.shape[0]),
        np.random.randint(source.shape[1]),
    )
    # # Black out image in-place
    destination[:]=source
    destination[np.maximum(coord[0]-crop_size, 0):np.minimum(coord[0]+crop_size, source.shape[0]-1), np.maximum(coord[1]-crop_size, 0):np.minimum(coord[1] + crop_size, source.shape[1]-1)] = np.array((0,0,0))
    
@njit(parallel=False, fastmath=True, inline='always')
def solarize_add(source, scratch, addition, destination):
    invert(source, scratch)
    scratch[:] = source + addition
    destination[:] = np.where(source >= np.array(128), source, scratch)

if __name__ == "__main__":
    import time
    import numpy as np
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
    from mpl_toolkits.axes_grid1 import ImageGrid

    # import pytest
    import math
    BATCH_SIZE = 512
    IMG_SIZE = 224
    image_pipelines = {
        'with': [CenterCropRGBImageDecoder((224, 224), ratio=224/256), RandAugmentV2(2, 10), ToTensor()],
    }
    #RandAugment(num_ops=2, magnitude=15.0), 
    for name, pipeline in image_pipelines.items():
        loader = Loader('/fsx/pingchiang/imagenet_ffcv_100class/val_400_0.1_90.ffcv', batch_size=BATCH_SIZE,
                        num_workers=2, order=OrderOption.RANDOM,
                        drop_last=True, pipelines={'image': pipeline})

        import matplotlib.pyplot as plt
        idx = 0
        
        for ims, labs in loader:
            fig = plt.figure(figsize=(10., 10.))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
            for ax, im in zip(grid, ims):
                ax.imshow(im)
            plt.show()
            plt.savefig('temp.png')
            import pdb;pdb.set_trace()
