from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

from glob import glob

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

import numpy as np
import cv2


def normalize(x):
    return (x - x.mean() ) / x.std()

def hist_eq(x):
    return cv2.equalizeHist(x)
    
def border_pad(shape, x):
    
    h, w = x.shape

    x = np.pad(x, ((0, shape - h), (0, shape - w)))
    
    return x

@curry
def fix_ratio(shape, x):
    
    h, w = x.shape

    if h <= w:
        ratio = h/w # smaller than 1
        h_ = shape
        w_ = round(h_ / ratio) # h_ > w_ 
    else:
        ratio = w/h
        w_ = shape
        h_ = round(w_ / ratio)

    x = cv2.resize(x, dsize=(h_, w_), interpolation=cv2.INTER_LINEAR)
    
    # x = border_pad(shape, x)
    
    return x

def to_RGB(x):    
    return np.array([x,x,x])

if __name__ == "__main__":
    
    from matplotlib import pyplot as plt
    
    image = plt.imread('./datasets/stanford/train/patient00001/study1/view1_frontal.jpg')
    
    image = hist_eq(image)
    image = normalize(image)
    image = fix_ratio(512)(image)
    
    print(image.shape)
    plt.imshow(image)

