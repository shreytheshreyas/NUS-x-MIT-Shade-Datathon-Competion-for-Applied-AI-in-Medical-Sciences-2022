from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from operator import methodcaller

import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T

try    : from .dataset_chexpert import Chexpert
except : from  dataset_chexpert import Chexpert

try    : from .dataset_mimic import Mimic
except : from  dataset_mimic import Mimic


def DATALOADER(data_path, labels, source, mode, batch_size, num_workers, shape):
    
    augment = augment_fn(mode, shape)
        
    dataset = {"chexpert" : lambda : Chexpert(data_path, augment, labels, mode),
               "mimic"    : lambda : Mimic(data_path, augment, labels, mode)}[source]
        
    return DataLoader(dataset(),
                      batch_size  = batch_size,
                      shuffle     = (mode == "train"),
                      pin_memory  = (mode == "train"),
                      num_workers = num_workers)
        
def augment_fn(mode, shape):
            
    if mode == "train":
        return T.Compose(
                    [T.ToPILImage(),                     
                     T.RandomResizedCrop(size = (shape, shape), scale=(0.95, 1.05), ratio=(0.9, 1.1)),
                     T.RandomAffine(degrees = (-10, 10), translate = (0.05, 0.05), scale = (0.95, 1.05)),
                     T.RandomHorizontalFlip(p=0.2),
                     T.RandomRotation((-15,15)),
                     T.ToTensor(),
                     T.Normalize(0.5, 0.5),
                     # T.Sharpness([0, 30]),
                    # T.Blurriness(), T.Noise([0., 0.05]),
                    # T.Brightness(),
                    # T.Rotation(), T.Scale([0.7, 1.3]),
                    ])
    else:
        return T.Compose(
                    [T.ToPILImage(), 
                     T.Resize(size = (shape, shape)),                     
                     T.ToTensor(),
                     T.Normalize(0.5, 0.5),
                    ])

if __name__ == "__main__":
    
    from matplotlib import pyplot as plt
    from torch.utils.data import ConcatDataset
    from tqdm import tqdm
        
    loader = DATALOADER(data_path   = './datasets/mit/',
                        labels      = ['no finding', 'enlarged cardiomediastinum', 'cardiomegaly',
                                       'airspace opacity', 'lung lesion', 'edema', 'consolidation',
                                       'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion',
                                       'pleural other', 'fracture', 'support devices'],
                        mode        = ["train", "valid", "test"][0],
                        source      = "mimic", 
                        shape       = 256,
                        batch_size  = 32,
                        num_workers = 12)
    
    for i, (imgs, labels) in enumerate(loader) :
        
        print(f"iteration : {i}")
        print(f"input shape : {imgs.shape}")                
        print(f"label shape : {labels.shape}")
        
        # an instance from batch
        img   = imgs[0]
        label = labels[0]
        print(label)
        plt.imshow(img.permute(1,2,0))
        plt.title(f"label:{label}")
        plt.axis('off')
        plt.show()
        # break
