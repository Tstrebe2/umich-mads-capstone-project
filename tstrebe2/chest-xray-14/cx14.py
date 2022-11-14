import os
import cx14
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchmetrics
import pytorch_lightning as pl

from PIL import Image

# define the LightningModule which is similar to torch.nn.Module
class Densenet121(pl.LightningModule):
    def __init__(self,
                 learning_rate:float=1e-3, 
                 momentum:float=.9, 
                 weight_decay:float=1e-4, 
                 class_weights=None, 
                 freeze_features:str='False',
                 lr_scheduler_patience:float=1e-4,
                 lr_scheduler_factor:float=.5,
                 lr_scheduler_min_lr:float=1e-5,
                 input_channels:int=1, 
                 out_features:int=15):

        super().__init__()
        self.automatic_optimization = True
        self.freeze_features = freeze_features
        self.f1_score = torchmetrics.F1Score(num_classes=out_features, average='macro')
        # This line saves the hyper_parameters so they can be called using self.hparams...
        self.save_hyperparameters('learning_rate', 
                                  'momentum', 
                                  'weight_decay', 
                                  'class_weights',
                                  'freeze_features',
                                  'lr_scheduler_patience',
                                  'lr_scheduler_factor',
                                  'lr_scheduler_min_lr')

        densenet = torchvision.models.densenet121(weights=None)
        
        self.features = densenet.features
        self.features.conv0 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classifier = torch.nn.Linear(in_features=densenet.classifier.in_features, 
                                          out_features=out_features, 
                                          bias=True)
        
    def forward(self, x):
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
        
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        
        class_weights = self.hparams.class_weights.to(self.device)
        train_loss = torch.nn.functional.cross_entropy(outputs, targets.squeeze(), weight=class_weights)
        
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        preds = torch.nn.functional.softmax(outputs, dim=1)
        
        class_weights = self.hparams.class_weights.to(self.device)
        val_loss = torch.nn.functional.cross_entropy(outputs, targets.squeeze(), weight=class_weights)
        
        val_f1_score = self.f1_score(preds, targets.squeeze())
        
        self.log_dict({ "val_loss":val_loss, "val_f1_score":val_f1_score }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        
        class_weights = self.hparams.class_weights.to(self.device)
        test_loss = torch.nn.functional.cross_entropy(outputs, targets.squeeze(), weight=class_weights)
        
        self.log("test_loss", on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self): 
        # We must store boolean values as strings. Pytorch Lightning has a
        # bug that changes boolean values when using DDP.
        if self.hparams.freeze_features == 'True':
            for param in self.features.parameters():
                param.requires_grad=False
            optimizer = torch.optim.SGD(self.classifier.parameters(), 
                                        lr=self.hparams.learning_rate, 
                                        momentum=self.hparams.momentum, 
                                        weight_decay=self.hparams.weight_decay)
        else:
            for param in self.parameters():
                if not param.requires_grad:
                    param.requires_grad=True
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.hparams.learning_rate, 
                                        momentum=self.hparams.momentum, 
                                        weight_decay=self.hparams.weight_decay)
        
        frozen_layers, non_frozen_layers = (0, 0)
        
        for param in self.parameters():
            if param.requires_grad:
                non_frozen_layers += 1
            else:
                frozen_layers +=1
        
        print('Training with {:,} frozen layers and {:,} non-frozen layers'.format(frozen_layers, non_frozen_layers))
            
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                optimizer,
                mode='min',
                patience=self.hparams.lr_scheduler_patience,
                factor=self.hparams.lr_scheduler_factor,
                min_lr=self.hparams.lr_scheduler_min_lr,
                verbose=True)
            
        return {"optimizer": optimizer,
                "lr_scheduler": 
                   {"scheduler": lr_scheduler,
                    "interval":"epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                    "strict":True},
                }

class CX14Dataset(Dataset):
    def __init__(self, img_dir, img_targets, transform=None):
        """
        :param image_dir: Directory where chest-xray images are stored.
        :param img_targets: Pandas dataframe of file names and target values.
        :param transform: function or squences of functions to apply transformations.
        """
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