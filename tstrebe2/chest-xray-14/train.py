import os
from functools import partial
import cx14

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer

from PIL import Image

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import socket

os.environ['MASTER_ADDR'] = socket.gethostname()

image_dir = '/home/tstrebel/assets/chest-xray-14/images/images'
target_dir = '/home/tstrebel/assets/chest-xray-14'

target_map = {'No Finding':0, 'Atelectasis':1, 'Cardiomegaly':2, 'Consolidation':3, 'Edema':4,
       'Effusion':5, 'Emphysema':6, 'Fibrosis':7, 'Hernia':8, 'Infiltration':9,
       'Mass':10, 'Nodule':11, 'Pleural_Thickening':12, 'Pneumonia':13, 'Pneumothorax':14}

train_val_df = pd.read_csv(os.path.join(target_dir, 'train_val_list.txt'), header=None, index_col=0).index
target_df = pd.read_csv(os.path.join(target_dir, 'Data_Entry_2017.csv'), usecols=[0, 1])
target_df.columns = ['file_path', 'target']
target_df = target_df[(target_df['file_path'].isin(train_val_df)) & ~(target_df.target.str.contains('\|'))]
target_df.target = target_df.target.map(target_map)
target_map = list(target_map.keys())
del(train_val_df)

X_train, X_val = train_test_split(target_df, stratify=target_df.target, test_size=.2, random_state=99)
X_val, X_test = train_test_split(X_val, stratify=X_val.target, test_size=.4, random_state=99)
train_ix, val_ix, test_ix = list(X_train.index), list(X_val.index), list(X_test.index)
del(X_train)
del(X_val)
del(X_test)

mean = [0.5341]
std = [0.2232]

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation((-2, 2)),
    torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1),
    torchvision.transforms.Resize(512),
    torchvision.transforms.CenterCrop(448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(512),
    torchvision.transforms.CenterCrop(448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

train_dataset = cx14.CX14Dataset(image_dir, target_df.loc[train_ix], num_classes=len(target_map), transform=train_transform)
val_dataset = cx14.CX14Dataset(image_dir, target_df.loc[val_ix], num_classes=len(target_map), transform=val_transform)

train_loader = cx14.get_data_loader(train_dataset, batch_size=22, shuffle=True)
val_loader = cx14.get_data_loader(val_dataset, batch_size=22)

model = cx14.Densenet121()

checkpoint_cb = pl.callbacks.ModelCheckpoint(dirpath='/home/tstrebel/models/',
                                             filename='cx14-densenet.pt',
                                             monitor='val_loss',
                                             verbose=True,
                                             save_top_k=-1,
                                             mode='min',
                                             save_weights_only=False)
freeze_unfreeze_cb = cx14.FeaturesFreezeUnfreeze()
early_stopping_cb = pl.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=1e-3,
                                               patience=5,
                                               verbose=True,
                                               mode='min',
                                               check_finite=True)
lr_monitor_cb = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

trainer = Trainer(accelerator='gpu', 
                  devices=-1,
                  num_nodes=2,
                  callbacks=[checkpoint_cb, freeze_unfreeze_cb, early_stopping_cb, lr_monitor_cb],
                  logger=logger,
                  max_epochs=2)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)