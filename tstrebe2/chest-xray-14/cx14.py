import os
import cx14

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=3, min_lr=5e-5, factor=0.75):
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

# define the LightningModule
class Densenet121(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, momentum=.9, weight_decay=1e-4, input_channels=1, out_features=15):
        super().__init__()
        self.save_hyperparameters('learning_rate', 'momentum', 'weight_decay')
        
        self.input_channels=input_channels
        self.out_features=out_features
        
        densenet = torchvision.models.densenet121(weights='DEFAULT')
        self.features = densenet.features
        self.features.conv0 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classifier = torch.nn.Linear(in_features=densenet.classifier.in_features, 
                                    out_features=self.out_features, 
                                    bias=True)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = None
        
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
        loss = self.criterion(outputs, targets)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        inputs, targets = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, targets)
        self.lr_scheduler(val_loss)
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        inputs, targets = batch
        outputs = self(inputs)
        test_loss = self.criterion(outputs, targets)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), 
                                    lr=self.hparams.learning_rate, 
                                    momentum=self.hparams.momentum, 
                                    weight_decay=self.hparams.weight_decay)
        self.lr_scheduler = LRScheduler(optimizer)
        return optimizer
    
class FeaturesFreezeUnfreeze(pl.callbacks.BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(pl_module.features)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
         # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor,
                optimizer=optimizer,
                train_bn=True)
            
class FineTuneBatchSizeFinder(pl.callbacks.BatchSizeFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.scale_batch_size(trainer, pl_module)
    
class CX14Dataset(Dataset):
    def __init__(self, img_dir, img_targets, num_classes, transform=None):
        self.img_targets = img_targets

        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.img_targets)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_targets.iloc[idx, 0])

        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        targets = np.array([self.img_targets.iloc[idx, -1]])
        targets = torch.from_numpy(targets)
        targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).squeeze().float()

        return image, targets
    
def get_data_loader(dataset, batch_size, shuffle=False, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)