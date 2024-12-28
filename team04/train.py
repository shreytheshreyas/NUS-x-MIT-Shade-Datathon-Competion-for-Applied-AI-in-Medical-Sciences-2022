import os, sys, time, warnings

warnings.filterwarnings('ignore')

from addict import Dict

from glob import glob
import argparse
from argparse import ArgumentParser

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from data.dataloader import DATALOADER
from models.baseline import NETS

from utils import COMPUTE_METRIC, COMPUTE_LOSS, GET_OPTIMIZER, COMPUTE_WEIGHTS

def parse():
    
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)    

    parser.add_argument("--data_path", type=str, default="./data/datasets/stanford-chexpert")
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--ckpt_path", type=str, default="./ckpts")
    parser.add_argument("--source", type=str, default="chexpert", help = "chexpert | mimic")    
    parser.add_argument("--additional", type=str, default="u0")
    
    parser.add_argument("--encoder_name", type=str, default="AlexNet", help = "AlexNet | VGGNet | ResNet | ViT | ConvNext") # in order of increasing performance
    parser.add_argument("--input_size", type=int, default=224)    
    parser.add_argument("--code_size", type=int, default=512)    
    
    parser.add_argument("--optimizer_name", type =str, default="Adam", help = "SGD | Adadelta | Adagrad | Adam | RMSprop")
    
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
                
    parser.add_argument("--gpu_n", type=str, default= "1")
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--num_workers", type=int, default = 8)
    parser.add_argument("--prec", type=int, default=32)
    
    # get args
    args = first(parser.parse_known_args())
    
    # pick a gpu that has the largest space
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_n
        
    args.labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                   'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                   'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                   'Pleural Other', 'Fracture', 'Support Devices']

    args.weights = COMPUTE_WEIGHTS(args.data_path, args.labels)                
    args.weights = torch.from_numpy(np.array(args.weights, dtype=np.float32)).cuda()
    
    return args


class Trainer(pl.LightningModule):

    def __init__(self, args):
        super(Trainer, self).__init__()
        
        self.args = args
        self.net  = NETS(args.encoder_name, [args.input_size]*2, len(args.labels), args.code_size)
        
    def forward(self, volumes):
        return self.net(volumes)
    
    def _step(self, b):

        self.b = b

        Xs, ys = b

        logits = self.forward(Xs)
        
        loss = COMPUTE_LOSS(logits, ys, self.args.weights)
            
        return loss, ys, logits
    
    def training_step(self, b, _):

        loss, ys, logits = self._step(b)

        self.log(f"TRAIN_LOSS", loss, on_epoch=True, prog_bar=True, logger=True)  

        return loss

    def eval_step(self, b, bi, prefix):
                    
        loss, ys, logits = self._step(b)
                
        return {f"{prefix}_LOSS_STEP"  : loss,
                f"{prefix}_TRUE_STEP"  : ys,
                f"{prefix}_LOGIT_STEP" : logits}
    
    def validation_step(self, b, bi):
        return self.eval_step(b, bi, "VALID")

    def test_step(self, b, bi):
        return self.eval_step(b, bi, "TEST")
        
    def validation_epoch_end(self, outs):
        
        true  = compose(torch.cat, list, map(get("VALID_TRUE_STEP")))(outs).long()
        preds = compose(torch.cat, list, map(get("VALID_LOGIT_STEP")))(outs)
        
        metric = COMPUTE_METRIC(preds, true)
        
        for i, diagnosis in enumerate(self.args.labels):
            self.log(f"VALID_AUC_{diagnosis}", metric["aucs"][i], logger=True)
            self.log(f"VALID_ACC_{diagnosis}", metric["accs"][i], logger=True)
        
        self.log("VALID_GLOBAL_AUC", np.mean(np.array(metric["aucs"])))
        self.log("VALID_GLOBAL_ACC", np.mean(np.array(metric["accs"])))
    
        return {"VALID_GLOBAL_AUC" : np.mean(np.array(metric["aucs"])),
                "VALID_GLOBAL_ACC" : np.mean(np.array(metric["accs"])),
                
                "VALID_AUCs" : metric["aucs"],
                "VALID_ACCs" : metric["accs"] }
    
    
    def configure_optimizers(self):
                
        optimizer = GET_OPTIMIZER(self.parameters(),
                                  self.args.optimizer_name,
                                  self.args.learning_rate,
                                  self.args.weight_decay)
        
        def lr_foo(epoch):
            
            if epoch < 3 : lr_scale  = 1
            else         : lr_scale  = 1
                
            return lr_scale

        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )

        return [optimizer], [scheduler]
    
if __name__ == '__main__':
    
    # get args and config
    args = parse()
            
    # dataloaders
    ###################################################
    train_loader = DATALOADER(data_path   = args.data_path,
                              labels      = args.labels,
                              source      = args.source, 
                              shape       = args.input_size,
                              batch_size  = args.batch_size,
                              num_workers = args.num_workers,
                              mode        = "train")
    
    valid_loader = DATALOADER(data_path   = args.data_path,
                              labels      = args.labels, 
                              source      = args.source, 
                              shape       = args.input_size,
                              batch_size  = args.batch_size,
                              num_workers = args.num_workers,
                              mode        = "valid")
        
    test_loader  = DATALOADER(data_path   = args.data_path,
                              labels      = args.labels, 
                              source      = args.source, 
                              shape       = args.input_size,
                              batch_size  = args.batch_size,
                              num_workers = args.num_workers,
                              mode        = "test")        
    
    # tensorBoard Logger
    ###################################################      
    log_save_dir = f'{args.log_path}/{args.encoder_name}'
    logger_name  = f"model:{args.encoder_name}|{args.additional}"
    tb_logger = pl_loggers.TensorBoardLogger(log_save_dir, logger_name, 0)
    
    # callbacks
    ###################################################          
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    
    callback_save_dir = f'{args.ckpt_path}/{args.encoder_name}'
    callback_name     = f"model:{args.encoder_name}|{args.additional}"
    
    ckpt_callback = ModelCheckpoint(
        dirpath    = callback_save_dir,
        filename   = callback_name,
        monitor    = 'VALID_GLOBAL_AUC',
        save_top_k = 1,
        verbose    = True,            
        mode       = 'max')
        
    # INIT TRAINER
    model = Trainer(args)
    
    trainer = pl.Trainer(gpus       = 1,
                         #accelerator='dp',
                         max_epochs = args.epoch,
                         logger     = tb_logger,                         
                         callbacks  = [ckpt_callback, lr_callback],
                         num_sanity_val_steps=1,
                         precision  = args.prec)
    
    trainer.fit(model, train_loader, valid_loader)
    
    #auto-loads the best weights from the previous run
    trainer.test(dataloaders=test_dataloader)
    
    
    
    
