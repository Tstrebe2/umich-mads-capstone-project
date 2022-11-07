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
    def __init__(self, patience=5, min_delta=0.001):
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

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=2, min_lr=5e-5, factor=0.1):
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
                verbose=True)
        
    def __call__(self, val_loss):
        # take one step of the learning rate scheduler while providing the validation loss as the argument
        self.lr_scheduler.step(val_loss)
        
def print_epoch_loss(phase, epoch_loss, epoch_acc, lr, since):
    time_elapsed = time.time() - since
    print('{} Loss: {}{:.4f} Acc: {:.4f} LR: {:6f} Time elapsed: {:.0f}m {:.0f}s'
          .format(phase, 
                  '\t' if phase == 'train' else '', 
                  epoch_loss, 
                  epoch_acc, 
                  lr,
                  time_elapsed // 60, 
                  time_elapsed % 60))
        
def train_model(model, 
                model_save_path, 
                train_dataset, 
                val_dataset,
                optimizer, 
                criterion,
                device,
                batch_size=16, 
                num_epochs=5,
                init_best_loss=np.inf,
                init_best_acc=0.0):
    since = time.time()

    best_acc = init_best_acc
    best_loss = init_best_loss
    
    train_dataloader = get_data_loader(train_dataset, batch_size, shuffle=True)
    val_dataloader = get_data_loader(val_dataset, batch_size, shuffle=False)
    
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()
    
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
                train_loss += loss.item() * inputs.size(0)
                train_corrects += torch.sum(preds == targets.data).item()
                
        with torch.no_grad():
                
            lr = next(iter(optimizer.param_groups))['lr']
            
            train_loss /= len(train_dataset)
            train_acc = train_corrects / len(train_dataset)

            print_epoch_loss('train', train_loss, train_acc, lr, since)
        
            model.eval()
            for inputs, targets in val_dataloader:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                preds = (outputs >= 0.0).float()
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == targets.data).item()
                
            val_loss /= len(val_dataset)
            val_acc = val_corrects / len(val_dataset)
                
            print_epoch_loss('validation', val_loss, val_acc, lr, since)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_acc = val_acc
                torch.save(
                    dict(model=model,
                         best_loss=best_loss,
                         best_acc=best_acc,
                         lr=lr,), 
                    model_save_path)
                
            lr_scheduler(val_loss)
            early_stopping(val_loss)
            
            if early_stopping.early_stop:
                print("Early stopping after epoch: {}/{}...".format(epoch+1, num_epochs),
                      "Loss: {:.6f}...".format(train_loss),
                      "Val Loss: {:.6f}".format(val_loss))            
                break
                
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f}')
    print(f'Best val Acc: {best_acc:4f}')
    
    
def load_checkpoint(model_save_path, device):
    checkpoint = torch.load(model_save_path, map_location=device)
    model = checkpoint['model']
    best_loss = checkpoint['best_loss']
    best_acc = checkpoint['best_acc']
    lr = checkpoint['lr']
    return model, best_loss, best_acc, lr
                