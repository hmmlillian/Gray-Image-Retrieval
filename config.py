"""config system.
"""

import os
import os.path as osp
import numpy as np

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Pixel mean values (BGR order) as a (1, 1, 3) array
# [0, 0, 0] for VGG trained with batch normalization
__C.PIXEL_MEANS = np.array([[[0, 0, 0]]])

# Pixel size of an image's shortest and longest side and input size for vgg classification
__C.TEST_MIN_SIZE = 256
__C.TEST_MAX_SIZE = 1000
__C.TEST_DATA_SIZE = 224

# Feature layer from which to extract
__C.FC_LAYER = 'fc6/bn'
__C.POOL_LAYER = 'roi_pool5_4'
__C.POOL_FTR_LEN = 7 # 7 x 7 pool5 array

# weight to balance feature difference and histogram difference
__C.FTR_WEIGHT = 1.0
__C.HIST_WEIGHT = 0.25