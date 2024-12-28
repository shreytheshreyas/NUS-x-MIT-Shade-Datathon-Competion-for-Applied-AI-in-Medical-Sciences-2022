import argparse

from toolz import *
from toolz.curried import *
import torch.nn as nn
import torch
import matplotlib.pyplot as plt 

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


try    : from  baseline import NETS
except : from .baseline import NETS

import sys, os

import numpy as np 

import cv2


sys.path.append("/root/team04/data/")

from dataloader import DATALOADER , augment_fn

from dataset_chexpert import Chexpert

#from dataloader import DATALOADER 
print(sys.path)

parser = argparse.ArgumentParser()

parser.add_argument("--model-path"  , default = "/root/team04/ckpts/AlexNet/model:AlexNet.ckpt")


if __name__ == '__main__':

    parse = parser.parse_args()

    checkpoint = torch.load(parse.model_path , map_location = lambda storage, loc:storage)['state_dict']

    input_size   = [224, 224]
    encoder_name = ["AlexNet","VGGNet","ResNet", "ConvNext", "ViT"][0]
    label_n      = 14
    code_size    = 512

    model = NETS(encoder_name , input_size , label_n , code_size)
    checkpoint = keymap(lambda x : x.replace("net.", ""))(checkpoint)
    model.load_state_dict(checkpoint)


    #print(model)
    target_layers = [model.backbone[0][-1]]
    #target_layers = [model.classifier[-1]]


    cam = AblationCAM(model=model, target_layers=target_layers, use_cuda= False)
    

    #

    model_robust = NETS(encoder_name , input_size , label_n , code_size)
    checkpoint_robust = torch.load("/root/team04/ckpts/AleNet_win/model:AlexNet_win.ckpt" , map_location = lambda storage, loc:storage)['state_dict']
    checkpoint_robust = keymap(lambda x : x.replace("net.", ""))(checkpoint_robust)
    model_robust.load_state_dict(checkpoint_robust)
    
    #cam = HiResCAM(model=model, target_layers=target_layers, use_cuda= False)
    
    # loader = DATALOADER(data_path   = '/root/team04/data/datasets/stanford-chexpert',
    #                     labels      = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    #                                  'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    #                                  'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    #                                  'Pleural Other', 'Fracture', 'Support Devices'],
    #                     # labels = ['No Finding', ],
    #                     mode        = ["train", "valid", "test"][0],
    #                     shape       = 256,
    #                     batch_size  = 32,
    #                     num_workers = 12)

    
    # for i, (imgs, labels) in enumerate(loader):
    #     print(f"iteration : {i}")
    #     print(f"input shape : {imgs.shape}")                
    #     print(f"label shape : {labels.shape}")
        
    # #     # an instance from batch
    #     img   = imgs[0]
    #     label = labels[0]
    #     print(img)
    #     print(label)
    #     break


    gen = Chexpert(data_path = '/root/team04/data/datasets/stanford-chexpert',
                   augment   = augment_fn("test" ,224),
                   mode      = ["train", "valid", "test"][0])    
    
    #print(len(gen))

    for idx in range(len(gen)):        
        img, labels = gen.__getitem__(idx)

        if labels[7] == 1:
            print(img.shape)
            break        
        else:
            continue

    
    logits = model.forward(img.unsqueeze(0))

    print("logits is" , logits)

    print(torch.argmax(logits))
    

    targets = [ClassifierOutputTarget(7)]
    grayscale_cam = cam(input_tensor=img.unsqueeze(0), targets= targets)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(img.cpu().permute(1,2,0).detach().numpy() , grayscale_cam, use_rgb=False)


    print(visualization.shape)
    print(img.cpu().detach().numpy().reshape((224 , 224 , 3)))

    cv2.imwrite("/root/team04/models/input1.png" , img.cpu().detach().numpy().reshape((224 , 224 , 3)))     
    cv2.imwrite("/root/team04/models/output1.png" , visualization) 