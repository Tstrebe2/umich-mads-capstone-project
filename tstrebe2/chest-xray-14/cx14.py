import os
import cx14

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

# define the LightningModule
class Densenet121(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, momentum=.9, weight_decay=1e-4, class_weights=None, input_channels=1, out_features=15):
        super().__init__()
        self.automatic_optimization = True
        self.save_hyperparameters('learning_rate', 'momentum', 'weight_decay', 'class_weights')
        
        densenet = torchvision.models.densenet121(weights='DEFAULT')
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
        class_weights = self.hparams.class_weights.to(self.device)
        val_loss = torch.nn.functional.cross_entropy(outputs, targets.squeeze(), weight=class_weights)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        class_weights = self.hparams.class_weights.to(self.device)
        test_loss = torch.nn.functional.cross_entropy(outputs, targets.squeeze(), weight=class_weights)
        self.log("test_loss", test_loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.classifier.parameters(), 
                                    lr=self.hparams.learning_rate, 
                                    momentum=self.hparams.momentum, 
                                    weight_decay=self.hparams.weight_decay)
        return optimizer
#         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#                                                                   patience=3, 
#                                                                   factor=0.9, 
#                                                                   mode='min', 
#                                                                   verbose=True)
        
#         return {"optimizer": optimizer, 
#                 "lr_scheduler": {'scheduler':lr_scheduler, "monitor": "val_loss", 'interval':'epoch', 'frequency':1}}
    
class FeaturesFreezeUnfreeze(pl.callbacks.BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, lr_on_unfreeze=1e-4):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._lr_on_unfreeze = lr_on_unfreeze

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `features`
        print('freezing feature weights')
        self.freeze(pl_module.features, train_bn=False)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
         # When `current_epoch` is 10, features will start training.
        if current_epoch == self._unfreeze_at_epoch:
            print('unfreezing feature weights')
            for pg in optimizer.param_groups:
                pg['lr'] = self._lr_on_unfreeze
            self.unfreeze_and_add_param_group(
                modules=[pl_module.features],
                optimizer=optimizer,
                lr=self._lr_on_unfreeze,
                train_bn=False)
    
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