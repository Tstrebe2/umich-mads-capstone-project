import os
from functools import partial
import cx14

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from torchmetrics import F1Score

from PIL import Image

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
                    prog = 'Densenet trainer.',
                    description = 'This script trains a densenet model on the chest x-ray 14 dataset.',
                    epilog = 'For help append train.py with --help')

parser.add_argument('--batch_size', 
                    nargs='?', 
                    default=16, 
                    help='Enter batch size for train & val.', 
                    type=int, 
                    required=False)

parser.add_argument('--epochs', 
                    nargs='?', 
                    default=15, 
                    help='Enter the total number of training epochs.', 
                    type=int, 
                    required=False)

parser.add_argument('--learning_rate', 
                    nargs='?', 
                    default=1e-3, 
                    help='enter the learning rate.', 
                    type=float, 
                    required=False)

parser.add_argument('--momentum', 
                    nargs='?', 
                    default=.9, 
                    help='enter momentum.', 
                    type=float, 
                    required=False)

parser.add_argument('--weight_decay', 
                    nargs='?', 
                    default=1e-4, 
                    help='enter weight decay.', 
                    type=float, 
                    required=False)

parser.add_argument('--num_stop_rounds', 
                    nargs='?', 
                    default=7, 
                    help='enter number of rounds for no improvement to trigger early stopping.', 
                    type=int, 
                    required=False)

parser.add_argument('--num_frozen_epochs', 
                    nargs='?', 
                    default=10, 
                    help='Enter the # of epochs to train while feature hidden layers are frozen.', 
                    type=int, 
                    required=False)

parser.add_argument('--image_dir', 
                    nargs='?', 
                    default='data/chest-xray-14/images', 
                    help='Enter directory to training images.', 
                    required=False)

parser.add_argument('--target_dir', 
                    nargs='?', 
                    default='data/chest-xray-14', 
                    help='Enter directory to csv file with targets.', 
                    required=False)

parser.add_argument('--models_dir', 
                    nargs='?', 
                    default='models', 
                    help='Directory to save models.', 
                    required=False)

parser.add_argument('--fast_dev_run', 
                    nargs='?', 
                    default=0, 
                    help='Flag to run quick train & val debug session on 1 batch.',
                    type=int,
                    required=False)

parser.add_argument('--num_nodes', 
                    nargs='?', 
                    default=2, 
                    help='Number of nodes for training.',
                    type=int,
                    required=False)

parser.add_argument('--num_workers', 
                    nargs='?', 
                    default=4, 
                    help='Number of workers from each node to assign to the data loader.',
                    type=int,
                    required=False)

parser.add_argument('--restore_ckpt_path', 
                    nargs='?', 
                    default='models/cx14-densenet-best.ckpt', 
                    help='Path to checkpoint to resume training.', 
                    required=False)

args = parser.parse_args()

if args.restore_ckpt_path == 'None':
    restore_ckpt_path=None
else:
    restore_ckpt_path = args.restore_ckpt_path
    
target_map = {'No Finding':0, 'Atelectasis':1, 'Cardiomegaly':2, 'Consolidation':3, 'Edema':4,
       'Effusion':5, 'Emphysema':6, 'Fibrosis':7, 'Hernia':8, 'Infiltration':9,
       'Mass':10, 'Nodule':11, 'Pleural_Thickening':12, 'Pneumonia':13, 'Pneumothorax':14}

train_val_df = pd.read_csv(os.path.join(args.target_dir, 'train_val_list.txt'), header=None, index_col=0).index
target_df = pd.read_csv(os.path.join(args.target_dir, 'Data_Entry_2017_v2020.csv'), usecols=[0, 1])
target_df.columns = ['file_path', 'target']
#target_df = target_df[(target_df['file_path'].isin(train_val_df)) & ~(target_df.target.str.contains('\|'))]
#target_df.target = target_df.target.map(target_map)
#target_map = list(target_map.keys())
target_df = target_df[(target_df['file_path'].isin(train_val_df))]
target_df.target = target_df.target.apply(lambda x: 1 if 'Pneumonia' in x else 0)
del(train_val_df)

X_train, X_val = train_test_split(target_df, stratify=target_df.target, test_size=.2, random_state=99)
X_val, X_test = train_test_split(X_val, stratify=X_val.target, test_size=.4, random_state=99)
train_ix, val_ix, test_ix = list(X_train.index), list(X_val.index), list(X_test.index)
del(X_train)
del(X_val)
del(X_test)

class_weights = (1-target_df.loc[train_ix].target.value_counts() / len(target_df)).sort_index().round(2).values
class_weights = torch.from_numpy(class_weights).float()

mean = [0.5341]
std = [0.2232]

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation((-4, 4)),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
    torchvision.transforms.Resize(512),
    torchvision.transforms.CenterCrop(448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(512),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

train_dataset = cx14.CX14Dataset(args.image_dir, 
                                 target_df.loc[train_ix], 
                                 transform=train_transform)
val_dataset = cx14.CX14Dataset(args.image_dir, 
                               target_df.loc[val_ix], 
                               transform=val_transform)
test_dataset = cx14.CX14Dataset(args.image_dir, 
                               target_df.loc[test_ix], 
                               transform=val_transform)

train_loader = cx14.get_data_loader(train_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers, 
                                    shuffle=True)
val_loader = cx14.get_data_loader(val_dataset, 
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers)
test_loader = cx14.get_data_loader(test_dataset, 
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers)

model = cx14.Densenet121(learning_rate=args.learning_rate, 
                         momentum=args.momentum, 
                         weight_decay=args.weight_decay, 
                         class_weights=class_weights)

checkpoint_cb = pl.callbacks.ModelCheckpoint(dirpath=args.models_dir,
                                             filename='cx14-densenet',
                                             monitor='val_loss',
                                             save_top_k=2,
                                             save_last=True,
                                             verbose=True,
                                             mode='min',
                                             save_weights_only=False)

freeze_unfreeze_cb = cx14.FeaturesFreezeUnfreeze(unfreeze_at_epoch=args.num_frozen_epochs)

early_stopping_cb = pl.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=1e-3,
                                               patience=args.num_stop_rounds,
                                               verbose=True,
                                               mode='min',
                                               check_finite=True)

lr_monitor_cb = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(accelerator='gpu', 
                  devices=-1,
                  num_nodes=args.num_nodes,
                  callbacks=[checkpoint_cb, freeze_unfreeze_cb, early_stopping_cb, lr_monitor_cb],
                  logger=True,
                  max_epochs=args.epochs,
                  fast_dev_run=bool(args.fast_dev_run))

trainer.fit(model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=restore_ckpt_path)

trainer.test(model=model,
            dataloaders=test_loader,
            ckpt_path='best',
            verbose=True)