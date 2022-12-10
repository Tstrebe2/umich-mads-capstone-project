import os
import pandas as pd
from functools import partial
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
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
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx]['file_path'])

        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        targets = self.img_targets.iloc[idx]['target']
        
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
            torchvision.transforms.Resize(512),
            torchvision.transforms.CenterCrop(448),
            torchvision.transforms.RandomHorizontalFlip(.25),
            torchvision.transforms.RandomRotation((-4, 4)),
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1),
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