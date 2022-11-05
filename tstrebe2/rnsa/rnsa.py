from collections import OrderedDict

import torchvision
import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

import pydicom as dicom
from PIL import Image

import copy
import time

class Densenet121(torch.nn.Module):
    def __init__(self, densenet, input_channels=1, out_features=1):
        super(Densenet121, self).__init__()
        
        self.input_channels=input_channels
        self.out_features=out_features
        
        self.features = densenet.features
        self.features.conv0 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classifier = torch.nn.Linear(in_features=densenet.classifier.in_features, 
                                    out_features=self.out_features, 
                                    bias=True)
        
        torch.nn.init.xavier_uniform_(self.classifier.weight)
    
    def forward(self, x):
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class RNSADataset(Dataset):
    def __init__(self, img_dir, annotations_file_path, indices=None, transform=None, target_transform=None):
        if indices:
            self.img_labels = pd.read_csv(annotations_file_path).iloc[indices]
        else:
            self.img_labels = pd.read_csv(annotations_file_path)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = ''.join([self.img_dir, '/', self.img_labels.iloc[idx, 0], '.dcm'])

        image = dicom.dcmread(img_path)
        image = Image.fromarray(image.pixel_array)

        label = self.img_labels.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    
def get_data_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after patience epochs.
    """
    def __init__(self, patience=50, min_delta=0.001):
        """
        :param patience: how many epochs to wait before stopping when loss is not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True
# Example code         
# for i in range(len(train_loss)):

#     early_stopping(train_loss[i], validate_loss[i])
#     print(f"loss: {train_loss[i]} : {validate_loss[i]}")
#     if early_stopping.early_stop:
#         print("We are at epoch:", i)
#         break

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=3, min_lr=5e-5, factor=0.1):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        # take one step of the learning rate scheduler while providing the validation loss as the argument
        self.lr_scheduler.step(val_loss)
        
def print_epoch_loss(phase, running_loss, running_corrects, lr, dataset_length, since):
    with torch.no_grad():
        epoch_loss = running_loss / dataset_length
        epoch_acc = running_corrects / dataset_length
        
    time_elapsed = time.time() - since
    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} LR: {lr:6f} Time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
def train_model(model, model_save_path, train_dataset, val_dataset, optimizer, criterion, batch_size=16, num_epochs=5):
    since = time.time()

    best_acc = 0.0
    best_loss = np.inf
    
    train_dataloader = get_data_loader(train_dataset, batch_size, shuffle=True)
    val_dataloader = get_data_loader(val_dataset, batch_size, shuffle=False)
    
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*10)
        
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        
        model.train()
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                preds = (outputs >= 0.0).float()
                train_loss += loss.item() * inputs.shape(0)
                train_corrects += torch.sum(preds == targets.data)
                
        print_epoch_loss('train', train_loss, train_corrects, next(iter(optimizer.param_groups))['lr'], len(train_dataset), since)
        
        with torch.no_grad():
            model.eval()
            for inputs, targets in val_dataloader:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                preds = (outputs >= 0.0).float()
                val_loss += loss.item() * inputs.shape(0)
                val_corrects += torch.sum(preds == targets.data)
                
            print_epoch_loss('validation', val_loss, val_corrects, next(iter(optimizer.param_groups))['lr'], len(val_dataset), since)
                
            lr_scheduler(val_loss)
            early_stopping(val_loss)
            
            if early_stopping.early_stop:
                print("Early stopping after epoch: {}/{}...".format(epoch+1, num_epochs),
                      "Loss: {:.6f}...".format(train_loss),
                      "Val Loss: {:.6f}".format(val_loss))            
                break
                
            if val_loss < best_loss:
                torch.save(model, model_save_path)
                best_loss = val_loss
                best_acc = val_corrects/len(val_dataset)
                
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f}')
    print(f'Best val Acc: {best_acc:4f}')
                