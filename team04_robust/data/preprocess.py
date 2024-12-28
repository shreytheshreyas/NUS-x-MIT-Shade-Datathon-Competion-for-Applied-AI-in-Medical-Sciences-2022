"""
this is the data preprocessing step to fix the image ratio and histogram normalize images.
I need this to accelerate the dataloading speed.
"""
import os
from glob import glob
import pandas as pd
from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from operator import methodcaller
from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from utils import hist_eq, fix_ratio
from multiprocessing import Pool
import matplotlib
import cv2


def parse():
    
    parser = ArgumentParser()
    
    parser.add_argument("--csvPath",  type=str, default="/data/volume02/MIMIC_CXR/train.csv")
    parser.add_argument("--loadPath", type=str, default="/data/volume02/MIMIC_CXR/valid")
    parser.add_argument("--savePath", type=str, default="./datasets/mit/valid")        
    parser.add_argument("--doTest", type=int, default=bool(0))
    parser.add_argument("--doResize", type=int, default=bool(1))    
    parser.add_argument("--doHistEq", type=int, default=bool(0))    
    parser.add_argument("--imgSize", type=int, default=320)
    
    args = first(parser.parse_known_args())
    
    args.labels = ['no finding', 'enlarged cardiomediastinum', 'cardiomegaly',
                   'airspace opacity', 'lung lesion', 'edema', 'consolidation',
                   'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion',
                   'pleural other', 'fracture', 'support devices'] # here I follow the convention in mit.
    
    return args

def _process (args) :
    
    resize = partial (fix_ratio, args.imgSize) if ( args.doResize ) else identitiy     
    histeq = hist_eq if (args.doHistEq) else identity
    
    return compose(histeq, resize)
        
def process (args, file) :
    
    newFile = file.replace(args.loadPath, args.savePath)
    newFolder = "/".join(newFile.split("/")[:-1])
    
    processedImg = compose(_process(args), plt.imread)(file)
    
    if not os.path.exists(newFolder):
        os.makedirs(newFolder)        

    cv2.imwrite(newFile, processedImg)        
        
if __name__ == '__main__':
        
    args = parse()
    
    # preprocess csv
    csv = pd.read_csv(args.csvPath)    
    csv.columns = [x.lower() for x in csv.columns]    
    csv = csv.rename(columns = dict(zip(csv.columns[-14:], args.labels)) )
    csv.to_csv(args.csvPath) # overwrite
    
    # preprocess images
    files = glob(f"{args.loadPath}/*/*/*.jpg")
    files = files[: int(len(files) * 0.001)] if (args.doTest) else files    
    with Pool(24) as p:
        p.map(partial(process, args), files)
    
    
    
    
    
    
    
    
    