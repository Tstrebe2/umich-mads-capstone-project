import pandas as pd
import numpy as np
from functools import partial

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
import torchvision
import torch

import pydicom as dicom
from PIL import Image

MEAN = [0.5]
STD = [0.225]

def get_training_data_target_dict(path:str, drop_split_col=True) -> dict:
    target_df = pd.read_csv(path, index_col='index')
    
    df_train = target_df[target_df.split == 'train']
    df_val = target_df[target_df.split == 'val']
    df_test = target_df[target_df.split == 'test']
    
    if drop_split_col:
        df_train = df_train.drop('split', axis=1)
        df_val = df_val.drop('split', axis=1)
        df_test = df_test.drop('split', axis=1)
    
    return dict(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
    )

class RSNADataset(Dataset):
    def __init__(self, 
                 img_dir:str, 
                 img_targets:pd.DataFrame, 
                 transform:torchvision.transforms.Compose=None, 
                 target_transform:torchvision.transforms.Compose=None) -> None:
        """
        img_dir: Path to dicom image files.
        img_targets: Data frame containing dicom image file names and 
            target variables.
        transform: Sequential tranformations for image data.
        target_transform: Sequential tranformations for target tensors.
        """
        self.img_dir = img_dir
        self.img_targets = img_targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_targets)

    def __getitem__(self, idx) -> tuple:
        img_path = ''.join([self.img_dir, self.img_targets.iloc[idx]['patient_id'], '.dcm'])

        image = dicom.dcmread(img_path)
        image = Image.fromarray(image.pixel_array)
        
        if self.transform:
            image = self.transform(image)

        target = self.img_targets.iloc[idx]['target']
        
        if self.target_transform:
            target = self.target_transform(target)

        return image, target
    
class RSNAIndexedDataset(RSNADataset):
    def __init__(self, 
                 img_dir:str, 
                 img_targets:pd.DataFrame, 
                 transform:torchvision.transforms.Compose=None, 
                 target_transform:torchvision.transforms.Compose=None) -> None:
        super().__init__(img_dir, img_targets, transform, target_transform)
        
    def __getitem__(self, idx) -> tuple:
        img_path = ''.join([self.img_dir, self.img_targets.iloc[idx]['patient_id'], '.dcm'])

        image = dicom.dcmread(img_path)
        image = Image.fromarray(image.pixel_array)
        
        if self.transform:
            image = self.transform(image)

        target = self.img_targets.iloc[idx]['target']
        
        if self.target_transform:
            target = self.target_transform(target)

        return idx, image, target


def get_dataset(img_dir:str, df:pd.DataFrame, train:bool=False, indexed:bool=False) -> Dataset:
    '''
    img_dir: directory where rsna dataset image files are stored.
    df: dataframe containing image file names (patient_id) and target variables.
    train: whether to return dataset with training data transforms.
    indedex: 
        if True, returns dataframe indicies "iloc", model inputs and targets.
        if False, model inputs and targets.
    '''
    target_transform = torchvision.transforms.Compose([
        partial(torch.tensor, dtype=torch.float),
        partial(torch.unsqueeze, dim=0),
    ])
    
    if train:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512),
            torchvision.transforms.CenterCrop(448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation((-4, 4)),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(MEAN, STD),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512),
            torchvision.transforms.CenterCrop(448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(MEAN, STD),
        ])
    
    if indexed:
        return RSNAIndexedDataset(img_dir, df, transform, target_transform)
    else:
        return RSNADataset(img_dir, df, transform, target_transform)
    
def get_data_loader(dataset:Dataset, batch_size:int, shuffle:bool=False, num_workers:int=4, pin_memory:bool=True) -> None:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
