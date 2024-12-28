import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict

# from sklearn.metrics import \
#     roc_curve, precision_recall_curve, auc as get_auc

from torchmetrics.functional import auroc
from torchmetrics.functional import accuracy

from torch.optim import \
    SGD, Adadelta, Adagrad, Adam, RMSprop


def COMPUTE_METRIC(outputs, targets):
     
    outputs = outputs.detach().cpu()
    targets = targets.detach().cpu()
    
    aucs, accs = [], []
    for i in range(outputs.shape[1]):
        
#        fpr, tpr, _ = roc_curve(targets[:,i], outputs[:,i])        
#        precision, recall, _ = precision_recall_curve(targets[:,i], outputs[:,i])    
#        auc = get_auc(fpr, tpr)
#        acc = sum(targets[:,i] == outputs[:,i].round().type_as(targets)).item() / len(outputs[:,i])

        auc = auroc(outputs[:,i], targets[:,i], task = "binary")
        acc = accuracy(outputs[:,i], targets[:,i], task = "binary")
                
        aucs.append(auc)
        accs.append(acc)
        
    return {'aucs' : aucs,
            'accs' : accs}
        
    
def COMPUTE_LOSS(output, target, weights):
    
    # weighted BCE_logit loss \sum_{task \in tasks} ( -y * log(sigmoid(y_hat)) )    
    loss = 0    
    for t in range(target.shape[-1]):
        loss = loss + _COMPUTE_LOSS(output, target, t, weights)
    
    return loss/target.shape[-1]

    
        
def _COMPUTE_LOSS(output, target, index, weights):
    
    target = target[:, index].view(-1)
    output = output[:, index].view(-1)

    if target.sum() == 0:
        loss = torch.tensor(0., requires_grad=True).cuda()
        
    else:
        
        loss = F.binary_cross_entropy_with_logits(output,
                                                  target,
                                                  pos_weight = torch.Tensor([weights[index]]).cuda())
    return loss

def rank_loss(feature):
    feature = feature.permute(0, 2, 3, 1).contiguous().view(-1, feature.size())
    feature = feature[torch.randperm(len(feature))]
    U, S, V = torch.svd(feature)
    low_rank_loss_spinal = S[14]
    return low_rank_loss

def GET_OPTIMIZER(params, opt_name, learning_rate, weight_decay):
    
    return \
        {"SGD"      : lambda : SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay),
         "Adadelta" : lambda : Adadelta(params, lr=learning_rate, weight_decay=weight_decay),
         "Adagrad"  : lambda : Adagrad(params, lr=learning_rate, weight_decay=weight_decay),
         "Adam"     : lambda : Adam(params, lr=learning_rate, weight_decay = weight_decay),
         "RMSprop"  : lambda : RMSprop(params, lr=learning_rate, momentum=0.9)}[opt_name]()

import pandas as pd

def COMPUTE_WEIGHTS(data_path, labels):
    
    csv_path = f"{data_path}/train.csv"

    df = pd.read_csv(csv_path).fillna(0).replace(-1,1)
    df = df[labels]

    _weights = df.sum(axis=0) / df.shape[0]
    weights = list((1-_weights) / _weights )

    return weights 

if __name__ == '__main__':
            
    from torch import nn
    
    output = torch.sigmoid(torch.rand(10,5)-0.5)
    target = torch.randint(0,2, [10,5]).float()
    batch_weight = True
    weights = [1,1,1,1,1]
        
    print(COMPUTE_LOSS(output, target, weights))
    print(COMPUTE_METRIC(output, target))
                
