import torch
import numpy as np

from toolz         import *
from toolz.curried import *

import torch.nn as nn
import torch.nn.functional as F

try    : from  backbones import AlexNet, VGGNet, ResNet, ViT, ConvNext, win_norm
except : from .backbones import AlexNet, VGGNet, ResNet, ViT, ConvNext, win_norm

class NETS(nn.Module):
    
    def __init__(self, encoder_name, input_size, label_n, code_size):
        
        super(NETS, self).__init__()
        
        
        backbones = {
            ""
            "AlexNet"     : lambda : nn.Sequential(AlexNet.alexnet().features),
            "VGGNet"      : lambda : nn.Sequential(VGGNet.vgg16().features),
            "ResNet"      : lambda : nn.Sequential(*list(ResNet.resnet50(pretrained=True).children())[:-2]),
            "ViT"         : lambda : nn.Sequential(ViT.vit_b_16()),
            "ConvNext"    : lambda : nn.Sequential(ConvNext.convnext_small()),
            'AlexNet_win' : lambda: nn.Sequential(win_norm.WindowNorm2d.convert_WIN_model(AlexNet.alexnet().features))}

        self.encoder_name = encoder_name
        self.input_size   = input_size
        self.label_n      = label_n
        self.code_size    = code_size 
        
        self.backbone = backbones[self.encoder_name]()
            
        self.z_shape = self.get_shape()
        
        if self.encoder_name != "ViT":
            self.classifier = nn.Sequential(nn.Conv2d(self.z_shape[0], self.code_size, 1),
                                            nn.Flatten(),
                                            nn.Dropout(p=0.5),
                                            nn.Linear(self.code_size * self.z_shape[1] * self.z_shape[2], self.code_size),
                                            nn.ReLU(),
                                            nn.Linear(self.code_size, self.label_n))
        else:
            self.classifier = nn.Sequential(nn.Linear(self.z_shape[0], self.code_size),
                                            nn.Dropout(p=0.5),
                                            nn.ReLU(),
                                            nn.Linear(self.code_size, self.label_n))            
        
    def get_shape(self):
                
        with torch.no_grad():
            
            X = torch.Tensor(1,3, *self.input_size)
            Y = self.backbone(X)

        torch.cuda.empty_cache()

        return Y.shape[1:]

    def forward(self, x):
                
        z     = self.backbone(x) # [B, C, H, W]
        logit = self.classifier(z) # [B, Y] Y : label_number
        
        return logit
    
if __name__ == "__main__" :
                    
    input_size   = [256, 256]
    encoder_name = ["AlexNet","VGGNet","ResNet", "ConvNext", "ViT"][0]
    label_n      = 14
    code_size    = 512
    
    x = torch.randn(2, 3, *input_size).cuda()    
    
    net = NETS(encoder_name, input_size, label_n, code_size).cuda()
    
    logit = net(x)
