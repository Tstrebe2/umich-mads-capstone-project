import os
import pandas as pd
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
from PIL import Image

def get_training_data_target_dict(target_dir:str) -> dict:
    target_df = pd.read_csv(os.path.join(target_dir, 'Data_Entry_2017.csv'), usecols=[0, 1])
    target_df.columns = ['file_path', 'target']
    # This is the target criteria for our cohort
    criteria = {'Infiltration', 'Consolidation', 'Atelectasis', 'Pneumonia', 'No Finding'}
    # This is the criterion we're using for pneumonia
    pneumonia = {'Infiltration', 'Consolidation', 'Atelectasis', 'Pneumonia'}

    target = (target_df
          .target
          .apply(lambda row: row.split('|'))
          .explode()
          .rename('target'))
    target = target[target.isin(criteria)].isin(pneumonia).astype(int)
    target = target.groupby(target.index).first()
    
    target_df = target_df[target_df.index.isin(target.index)]
    
    target_df['target'] = target

    #Set random_state to 99 for reproduceability
    X_train, X_val = train_test_split(target_df, stratify=target_df.target, test_size=.2, random_state=99)
    X_val, X_test = train_test_split(X_val, stratify=X_val.target, test_size=.4, random_state=99)
    train_ix, val_ix, test_ix = list(X_train.index), list(X_val.index), list(X_test.index)
    
    data_dict = {
        'df_train':target_df.loc[train_ix],
        'df_val':target_df.loc[val_ix],
        'df_test':target_df.loc[test_ix],
    }
    
    return data_dict

class CX14Dataset(Dataset):
    def __init__(self, img_dir, img_targets, transform=None, target_transform=None):
        """
        :param image_dir: Directory where chest-xray images are stored.
        :param img_targets: Pandas dataframe of file names and target values.
        :param transform: function or squences of functions to apply transformations.
        """
        self.img_targets = img_targets
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_targets)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 0])

        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        targets = self.img_targets.iloc[idx, -1]
        
        if self.target_transform:
            targets = self.target_transform(targets)
        
        return image, targets
    
def get_dataset(img_dir:str, df:pd.DataFrame, train:bool=False) -> None:
    # Using single-channel mean and standard deviation of 15,000 image samples from the training set.
    mean = [0.5341]
    std = [0.2232]
    
    target_transform = torchvision.transforms.Compose([
        partial(torch.tensor, dtype=torch.float),
        partial(torch.unsqueeze, dim=0),
    ])
    
    if train:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation((-4, 4)),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
            torchvision.transforms.Resize(512),
            torchvision.transforms.CenterCrop(448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512),
            torchvision.transforms.CenterCrop(448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        
    return CX14Dataset(img_dir, df, transform, target_transform)
    
def get_data_loader(dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)