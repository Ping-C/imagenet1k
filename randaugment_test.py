
import torch
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import randaugment_tf
import randaugment_v2
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation, AllocationQuery
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.state import State



# load an PIL image
# img = Image.open('/fs/cml-datasets/ImageNet/ILSVRC2012/train/n01744401/n01744401_7667.JPEG')
# img = np.asarray(img)
# num_bins=30
# op_table = [
#     (1, "ShearX", np.linspace(0.1, 1.8*3+0.1, num_bins), -1, lambda src, dst, mag :randaugment_v2.shear(src, dst, -mag, 0), randaugment_tf.shear_x),
#     (2, "ShearY", np.linspace(0.1, 1.8*3+0.1, num_bins), -1, lambda src, dst, mag :randaugment_v2.shear(src, dst, 0, -mag), randaugment_tf.shear_y),
#     (3, "TranslateX", np.linspace(0.0, 300, num_bins), -1, lambda src, dst, mag : randaugment_v2.translate(src, dst, int(-mag), 0), randaugment_tf.translate_x),
#     (4, "TranslateY", np.linspace(0.0, 300, num_bins), -1, lambda src, dst, mag : randaugment_v2.translate(src, dst, 0, int(-mag)), randaugment_tf.translate_y),
#     (5, "Rotate", np.linspace(0.0, 90.0, num_bins), -1, randaugment_v2.rotate, randaugment_tf.rotate),
#     (6, "Brightness", np.linspace(0.1, 1.8*3+0.1, num_bins), 1, randaugment_v2.adjust_brightness, randaugment_tf.brightness),
#     (7, "Color", np.linspace(0.1, 1.8*3+0.1, num_bins), 1, randaugment_v2.adjust_saturation, randaugment_tf.color),
#     (8, "Contrast", np.linspace(0.1, 1.8*3+0.1, num_bins), 1, randaugment_v2.adjust_contrast, randaugment_tf.contrast),
#     (9, "Sharpness", np.linspace(0.1, 1.8*3+0.1, num_bins), 1,  randaugment_v2.sharpen, randaugment_tf.sharpness),
#     (10, "Posterize", 8 - (np.arange(num_bins) / ((num_bins - 1) / 4)).round(), 1, randaugment_v2.posterize, randaugment_tf.posterize),
#     (11, "Solarize", np.linspace(0, 255*3, num_bins), 1, randaugment_v2.solarize, randaugment_tf.solarize),
#     (12, "AutoContrast", np.array(0.0), 1, randaugment_v2.autocontrast, randaugment_tf.autocontrast),
#     (13, "Equalize", np.array(0.0), 1, randaugment_v2.equalize, randaugment_tf.equalize),
#     # (14, "CutOut", np.linspace(0, 120, num_bins), 1),
#     (15, "Invert",  np.array(0.0), 1, randaugment_v2.invert, randaugment_tf.invert),
#     (16, "SolarizeAdd", np.linspace(0, 110*3, num_bins), 1, randaugment_v2.solarize_add, randaugment_tf.solarize_add),
# ]
# fig, axes = plt.subplots(2*len(op_table), 5, figsize=(5*4, 2*len(op_table)*4))
# i = 0
# for row, (_, opname, levels, negate, ffcvop, tfop) in enumerate(op_table):
#     for li in range(3, 8):
#         try:
#             level = levels[li]
#         except:
#             level = None

#         img_ffcvaug = np.array(img)
#         if opname in ["Brightness", "Color", "Contrast", "Sharpness", "Solarize", "SolarizeAdd"]:
#             scratch = np.array(img)
#             ffcvop(img, scratch, level, img_ffcvaug)
#         elif opname == "Posterize":
#             ffcvop(img, int(level), img_ffcvaug)
#         elif opname == "AutoContrast":
#             scratch = np.array(img)
#             ffcvop(img, scratch, img_ffcvaug)
#         elif opname == "Invert":
#             ffcvop(img, img_ffcvaug)
#         elif opname == "Equalize":
#             scratch = np.zeros((3, 256), dtype=np.dtype('int32'))
#             ffcvop(img, scratch, img_ffcvaug)
#         else:
#             ffcvop(img, img_ffcvaug, level)
        
#         # augment with tf
#         try:
#             img_tfaug = tfop(img, level, replace=[128,128,128])
#         except:
#             try:
#                 img_tfaug = tfop(img, level)
#             except:
#                 img_tfaug = tfop(img)
#         axes[row*2][li-3].imshow(img_ffcvaug)
#         axes[row*2+1][li-3].imshow(img_tfaug)
#         diff = np.max(np.abs(np.array(img_ffcvaug - img_tfaug)))
#         diff_mean = np.mean(np.abs(np.array(img_ffcvaug - img_tfaug)))
#         print(row, li, opname, diff, diff_mean)
# plt.savefig('augment.png')

# check the levels
# (level/_MAX_LEVEL) * 0.3
num_bins = 31
op_table = [
            (1, "ShearX", np.linspace(0.0, 0.3*3, num_bins), -1),
            (2, "ShearY", np.linspace(0.0, 0.3*3, num_bins), -1),
            (3, "TranslateX", np.linspace(0.0, 300, num_bins), -1),
            (4, "TranslateY", np.linspace(0.0, 300, num_bins), -1),
            (5, "Rotate", np.linspace(0.0, 90.0, num_bins), -1),
            (6, "Brightness", np.linspace(0.1, 1.8*3+0.1, num_bins), 1),
            (7, "Color", np.linspace(0.1, 1.8*3+0.1, num_bins), 1),
            (8, "Contrast", np.linspace(0.1, 1.8*3+0.1, num_bins), 1),
            (9, "Sharpness", np.linspace(0.1, 1.8*3+0.1, num_bins), 1),
            (10, "Posterize", (np.arange(num_bins) / ((num_bins - 1) / 12)).round(), 1),
            (11, "Solarize", np.linspace(0, 256*3, num_bins), 1),
            (12, "AutoContrast", np.array(0.0), 1),
            (13, "Equalize", np.array(0.0), 1),
            (14, "CutOut", np.linspace(0, 120, num_bins), 1),
            (15, "Invert",  np.array(0.0), 1),
            (16, "SolarizeAdd", np.linspace(0, 110*3, num_bins), 1),
        ]
import dataclasses
@dataclasses.dataclass
class HParams:
  """Parameters for AutoAugment and RandAugment."""
  cutout_const: int
  translate_const: int
augmentation_hparams = HParams(
      cutout_const=40, translate_const=100)
level = 11
for i, opname, mags, neg in op_table:
    try:
        if opname == "CutOut":
            print(f"{opname:<20}{mags[level]}, {randaugment_tf.level_to_arg(augmentation_hparams)['Cutout'](level)}")
        else:
            print(f"{opname:<20}:{mags[level]}, {randaugment_tf.level_to_arg(augmentation_hparams)[opname](level)}")
    except:
        print(f"{opname:<20}:{mags}, {randaugment_tf.level_to_arg(augmentation_hparams)[opname](level)}")


import pdb;pdb.set_trace()
pass
# feed it through tensorflow augmentation and ffcv augmentation

# compare the differences

