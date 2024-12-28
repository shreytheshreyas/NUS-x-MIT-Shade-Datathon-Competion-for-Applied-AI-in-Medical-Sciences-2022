import torch
import argparse
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()


parser.add_argument("--model-path" , type = str , default = "/root/team04/ckpts/ViT/model:ViT.ckpt")
parser.add_argument("--data_path", type=str, default="./data/datasets/stanford-chexpert")
parser.add_argument("--gpu_n", type=str, default= "1")
parser.add_argument("--batch_size", type=int, default = 32)
parser.add_argument("--num_workers", type=int, default = 8)

args.labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                   'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                   'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                   'Pleural Other', 'Fracture', 'Support Devices']

                
if __name__ == '__main__':

    parse = parser.parse_args()
    print(parse)

    #PLOT ROC Curve

    test_loader  = DATALOADER(data_path   = args.data_path,
                            labels      = args.labels,                        
                            shape       = args.input_size,
                            batch_size  = args.batch_size,
                            num_workers = args.num_workers,
                            mode        = "test")    

