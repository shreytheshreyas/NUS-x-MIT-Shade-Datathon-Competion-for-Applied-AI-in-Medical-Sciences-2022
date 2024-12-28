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

class Mimic(Dataset):
    
    def __init__(self,
                 data_path = './datasets',
                 augment   = identity,
                 labels    = ['no finding', 'enlarged cardiomediastinum', 'cardiomegaly',
                              'airspace opacity', 'lung lesion', 'edema', 'consolidation',
                              'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion',
                              'pleural other', 'fracture', 'support devices'],
                 mode      = ["train", "valid", "test"][0]):
                    
        self.datapoints = parse(data_path, labels, mode)
        self.augment    = augment
        self.mode       = mode
        self.data_path  = data_path
        
    def __getitem__(self, i):
        
        datapoint = self.datapoints[i]
        
        img_path, labels = datapoint["path"], datapoint["labels"]
        
        return compose(self.augment, cv2.imread)(img_path), torch.tensor(labels)
        
    def __len__(self):        
        return len(self.datapoints)
    
def parse_patient_name(patient_path, segment_num):
    string_segments = patient_path.split('/')
    return string_segments[segment_num]    

def parse(data_path, labels, mode):


    view_name = "view"
    view_item = "frontal"        
    path_fn = lambda path : "/".join([data_path] + path.split("/"))
        
    # read and set to lower case
    
    if mode  in ["train", "test"]:
        df = pd.read_csv(f"{data_path}/train.csv")
    if mode == "valid":
        df = pd.read_csv(f"{data_path}/valid.csv")
    
    # only one view
    df = df[df[view_name] == view_item]
    
    # filter label & merge
    if not labels == []:
        df[labels] = df[labels].fillna(0).replace(-1,0)

    df["labels"] = df[labels].values.tolist()
    # add a patient column to aid split
    patients = set()
    # patient_lists = []
    for index, row in df.iterrows():
        current_patient = parse_patient_name(row['path'], 1)
        patients.add(current_patient)
    patients = list(patients)
            
    # parse path
    df["path"] = df["path"].apply(path_fn)

    # to list of dict
    out = df[["path","labels"]].to_dict("records")
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
        if parse_patient_name(r['path'], 1) in test_patient:
            test_out.append(r)
        else:
            train_out.append(r)
            
    if mode == "train" : return train_out
    if mode == "test"  : return train_out[int(0.7*len(train_out)):]
        

if __name__ == "__main__":
    
    gen = Mimic(data_path = './datasets/mit',
                augment   = identity,
                mode      = ["train", "valid", "test"][0])    
    
    for idx in range(len(gen)):        
        
        img, labels = gen.__getitem__(idx)
        
        print(img.shape)
        
