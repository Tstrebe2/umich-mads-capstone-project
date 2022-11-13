import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import pydicom as dicom

class RSNADataset(Dataset):
    def __init__(self, img_dir, img_targets, transform=None):
        self.img_targets = img_targets

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_targets)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 0])

        image = dicom.dcmread(img_path)
        image = Image.fromarray(image.pixel_array).convert(mode='L')

        if self.transform:
            image = self.transform(image)

        targets = np.array([self.img_targets.iloc[idx, -1]])
        targets = torch.from_numpy(targets)

        return image, targets

class CX14Dataset(Dataset):
    def __init__(self, img_dir, img_targets, transform=None):
        self.img_targets = img_targets

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_targets)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 0])

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        targets = np.array([self.img_targets.iloc[idx, -1]])
        targets = torch.from_numpy(targets)

        return image, targets
        
def get_data_loader(dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)