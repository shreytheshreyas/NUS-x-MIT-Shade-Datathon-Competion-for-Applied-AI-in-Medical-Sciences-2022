import torch
import torch.nn as nn
from pytorch_revgrad import RevGrad


"""
I give two options for the choice of FC layer.
    1. (MLP)    A standard FC layer
    2. (MLP_GR) FC layer with graident reversal.
"""

class STACK_MLP(nn.Module):
    
    def __init__(self, backbone, classN):
        super().__init__()
        
        self.backbone = backbone
        
        self.make_mlp = lambda classN : nn.Sequential(nn.Linear(backbone.zDim, backbone.zDim//2), nn.ReLU(),
                                                      nn.Linear(backbone.zDim//2, backbone.zDim//4), nn.ReLU(),
                                                      nn.Linear(backbone.zDim//4, classN))
        
        self.mlp = self.make_mlp(classN)
    
    def forward(self, x):        
        
        y = self.mlp(self.backbone(x))
        
        return {"pred":y}
    
class STACK_MLP_GR(nn.Module):
    
    def __init__(self, backbone, classN, demographics):
        super().__init__()
        
        self.backbone = backbone
        
        self.make_mlp = lambda classN : nn.Sequential(nn.Linear(backbone.zDim, backbone.zDim//2), nn.ReLU(),
                                                      nn.Linear(backbone.zDim//2, backbone.zDim//4), nn.ReLU(),
                                                      nn.Linear(backbone.zDim//4, classN))
        
        self.add_gr  = lambda layer : nn.Sequential(layer,RevGrad()) 
        
        
        self.mlp = self.make_mlp(classN)
        
        mlpGrs = {}
        for k,v in demographics.items():            
            mlpGrs[k] = self.add_gr(self.make_mlp(v))
        
        self.mlpGrs = nn.ModuleDict(mlpGrs)
    
    def forward(self, x):
        
        z = self.backbone(x)
    
        outcome = {}
        outcome["pred"] = self.mlp(z)    
        
        for k,v in self.mlpGrs.items():            
            outcome[k] = v(z)
            
        return outcome
        
if __name__ == "__main__":

    # unit test for all models
    from AlexNet  import AlexNet
    from VGGNet   import vgg16_in as VGGNet
    from ResNet   import resnet101 as ResNet
    from ConvNext import convnext_base as ConvNeXtNet 
    
    def test (backbone, RV = True):
        
        # check by output shape
        
        classN = 5
        demographics = {"gender"    : 2,
                        "ethnicity" : 10,
                        "age_group" : 5}
        
        X = torch.randn(32,3,256,256).to("cuda")
        
        if not RV:
            
            net = STACK_MLP(backbone, classN).to("cuda")
            
            Y = net(X)
            
        else:
            
            net = STACK_MLP_GR(backbone, classN, demographics).to("cuda")
            
            Y = net(X)
            
        for k,v in Y.items():            
            print(k, v.shape)
            
        return None
        
    test(AlexNet(), RV = True)
        
        
        
        



