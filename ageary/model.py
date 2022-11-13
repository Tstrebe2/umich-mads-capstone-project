import torch
import torchvision
import pytorch_lightning as pl
from torchmetrics import F1Score

class DenseNet121(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, momentum=.9, weight_decay=1e-4, class_weights=None, input_channels=1, out_features=2):
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
        self.log("train_loss", train_loss, sync_dist=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        class_weights = self.hparams.class_weights.to(self.device)
        val_loss = torch.nn.functional.cross_entropy(outputs, targets.squeeze(), weight=class_weights)
        self.log("val_loss", val_loss, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        class_weights = self.hparams.class_weights.to(self.device)
        test_loss = torch.nn.functional.cross_entropy(outputs, targets.squeeze(), weight=class_weights)
        test_f1 = F1Score(num_classes=2).to(self.device)
        test_f1 = test_f1(outputs, targets.squeeze())
        self.log("test_loss", test_loss, sync_dist=True)
        self.log("test_f1", test_f1, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.classifier.parameters()), 
                                    lr=self.hparams.learning_rate,
                                    momentum=self.hparams.momentum, 
                                    weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, min_lr=5e-5, factor=0.75, mode='min', verbose=True)
        
        return {"optimizer": optimizer, "lr_scheduler": {'scheduler':lr_scheduler, "monitor": "val_loss", 'interval':'epoch', 'frequency':1}}

class AlexNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, momentum=.9, weight_decay=1e-4, class_weights=None, input_channels=1, out_features=2):
        super().__init__()
        self.automatic_optimization = True
        self.save_hyperparameters('learning_rate', 'momentum', 'weight_decay', 'class_weights')
        
        alexnet = torchvision.models.alexnet(weights='DEFAULT')
        self.features = alexnet.features
        self.features.conv0 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classifier = torch.nn.Linear(in_features=alexnet.classifier.in_features, 
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
        self.log("train_loss", train_loss, sync_dist=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        class_weights = self.hparams.class_weights.to(self.device)
        val_loss = torch.nn.functional.cross_entropy(outputs, targets.squeeze(), weight=class_weights)
        self.log("val_loss", val_loss, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        class_weights = self.hparams.class_weights.to(self.device)
        test_loss = torch.nn.functional.cross_entropy(outputs, targets.squeeze(), weight=class_weights)
        test_f1 = F1Score(num_classes=2).to(self.device)
        test_f1 = test_f1(outputs, targets.squeeze())
        self.log("test_loss", test_loss, sync_dist=True)
        self.log("test_f1", test_f1, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.classifier.parameters()), 
                                    lr=self.hparams.learning_rate, 
                                    momentum=self.hparams.momentum, 
                                    weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, min_lr=5e-5, factor=0.75, mode='min', verbose=True)
        
        return {"optimizer": optimizer, "lr_scheduler": {'scheduler':lr_scheduler, "monitor": "val_loss", 'interval':'epoch', 'frequency':1}}

class FeaturesFreezeUnfreeze(pl.callbacks.BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `features`
        print('freezing feature weights')
        self.freeze(pl_module.features, train_bn=False)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
         # When `current_epoch` is 10, features will start training.
        if current_epoch == self._unfreeze_at_epoch:
            print('unfreezing feature weights')
            self.unfreeze_and_add_param_group(
                modules=[pl_module.features],
                optimizer=optimizer,
                train_bn=False)