import random

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from operator import methodcaller

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
import cv2
from torch.utils.data import Dataset

class Chexpert(Dataset):
    
    def __init__(self,
                 data_path = './datasets',
                 augment   = identity,
                 labels    = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                              'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                              'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                              'Pleural Other', 'Fracture', 'Support Devices'],
                 mode      = ["train", "valid", "test"][0]):
                    
        self.datapoints = parse(data_path, labels, mode)
        self.augment    = augment
        self.mode       = mode
        self.data_path  = data_path
        
    def __getitem__(self, i):
        
        datapoint = self.datapoints[i]
        
        img_path, labels = datapoint["Path"], datapoint["Labels"]
        
        return compose(self.augment, cv2.imread)(img_path), torch.tensor(labels)
        
    def __len__(self):
        
        return len(self.datapoints)

def parse_patient_name(patient_path, segment_num):
    string_segments = patient_path.split('/')
    return string_segments[segment_num]    

def parse(data_path, labels, mode):
    # if labels == ['No Finding', 'Pneumonia', 'Others']:

    view_name = "Frontal/Lateral"
    view_item = "Frontal"        
    path_fn = lambda path : "/".join([data_path] + path.split("/")[1:])
        
    if mode  in ["train", "test"]:
        df = pd.read_csv(f"{data_path}/train.csv")
    if mode == "valid":
        df = pd.read_csv(f"{data_path}/valid.csv")

    # only one view
    df = df[df[view_name] == view_item]
    
    # filter label & merge
    if not labels == []:
        df[labels] = df[labels].fillna(0).replace(-1,0)

    df["Labels"] = df[labels].values.tolist()
    # add a patient column to aid split
    patients = set()
    # patient_lists = []
    for index, row in df.iterrows():
        current_patient = parse_patient_name(row['Path'], 2)
        patients.add(current_patient)
    patients = list(patients)
            
    # parse path
    df["Path"] = df["Path"].apply(path_fn)

    # to list of dict
    out = df[["Path","Labels"]].to_dict("records")
    random.seed(1)
    random.shuffle(out)
    if mode == "valid" : return out

    # split based on patient
    random.shuffle(patients)
    train_patient = patients[:int(len(out) * 0.7)]
    test_patient = patients[int(len(out) * 0.7):]
    train_out = []
    test_out = []
    for r in out:
        if parse_patient_name(r['Path'], 1) in test_patient:
            test_out.append(r)
        else:
            train_out.append(r)
    if mode == "train" : return train_out
    if mode == "test"  : return test_out
    
    '''
    # to list of dict
    out = df[["Path","Labels"]].to_dict("records")
    
    #shuffle to remove possible order bias
    random.seed(1)
    random.shuffle(out)

    if mode == "train" : return out[: int(len(out) * 0.7)]
    if mode == "test"  : return out[  int(len(out) * 0.7) :]
    if mode == "valid" : return out
    '''

    
    

if __name__ == "__main__":
    
    gen = Chexpert(data_path = './datasets/stanford-chexpert',
                   augment   = identity,
                   mode      = ["train", "valid", "test"][0])    
    
    for idx in range(len(gen)):        
        
        img, labels = gen.__getitem__(idx)
        
        print(img.shape)
        
