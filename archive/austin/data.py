import pandas as pd
from functools import partial

from torch.utils.data import DataLoader, Dataset
import torchvision
import torch
import os
import pydicom as dicom
from PIL import Image

def get_training_data_target_dict(path:str) -> dict:
    target_df = pd.read_csv(path, index_col='index')

    df_train = target_df[target_df.split == 'train'].drop('split', axis=1)
    df_val = target_df[target_df.split == 'val'].drop('split', axis=1)
    df_test = target_df[target_df.split == 'test'].drop('split', axis=1)

    return dict(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
    )

class CustomDataset(Dataset):
    def __init__(self, 
                 img_dir:str, 
                 img_targets:pd.DataFrame, 
                 model:str,
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
        self.model = model

    def __len__(self) -> int:
        return len(self.img_targets)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 0])
        if '.' not in img_path:
            img_path = img_path + '.dcm'
            image = dicom.dcmread(img_path)
            image = Image.fromarray(image.pixel_array)
        else:
            image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)

        target = self.img_targets.iloc[idx, -1]
        
        if self.target_transform:
            target = self.target_transform(target)
        
        if self.model == 'resnet':
            image = image.repeat(3, 1, 1)
            
        return image, target

def get_dataset(img_dir:str, df:pd.DataFrame, model:str, train:bool=False) -> None:
    # Single channel mean & standard deviation.
    mean = [0.5]
    std = [0.225]
    
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
            torchvision.transforms.Normalize(mean, std),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512),
            torchvision.transforms.CenterCrop(448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        
    return CustomDataset(img_dir, df, model, transform, target_transform)

def get_data_loader(dataset:Dataset, batch_size:int, shuffle:bool=False, num_workers:int=4, pin_memory:bool=True) -> None:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
