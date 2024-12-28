from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from operator import methodcaller

import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T

try    : from .dataset import Chexpert
except : from  dataset import Chexpert

def DATALOADER(data_path, labels, mode, batch_size, num_workers, shape):
    
    augment = augment_fn(mode, shape)
        
    chexpert = lambda : Chexpert(data_path, augment, labels, mode)
        
    return DataLoader(chexpert(),
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
                     T.GaussianBlur(3),
                     T.ColorJitter(0, 0, 0, 0.25),
                     T.Normalize(0.5, 0.5),
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
        
    loader = DATALOADER(data_path   = './datasets/stanford-chexpert',
                        labels      = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                                     'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                                     'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                                     'Pleural Other', 'Fracture', 'Support Devices'],
                        # labels = ['No Finding', ],
                        mode        = ["train", "valid", "test"][0],
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
