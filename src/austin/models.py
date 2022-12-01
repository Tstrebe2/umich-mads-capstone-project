import torch
import torchvision
import pytorch_lightning as pl
import torchmetrics
import numpy as np


class DenseNet121(pl.LightningModule):
    def __init__(self,
                 learning_rate:float=1e-3, 
                 momentum:float=.9, 
                 weight_decay:float=1e-4, 
                 class_weights=None, 
                 freeze_features:str='False',
                 T_max:int=50,
                 eta_min:float=5e-5,
                 input_channels:int=1, 
                 out_features:int=1):

        super().__init__()
        self.automatic_optimization = True
        self.freeze_features = freeze_features
        self.average_precision = torchmetrics.AveragePrecision(num_classes=out_features, average=None)
        # This line saves the hyper_parameters so they can be called using self.hparams...
        self.save_hyperparameters('learning_rate', 
                                  'momentum', 
                                  'weight_decay', 
                                  'class_weights',
                                  'freeze_features',
                                  'T_max',
                                  'eta_min')
        
        densenet = torchvision.models.densenet121(weights='DEFAULT')
        
        self.features = densenet.features
        self.features.conv0 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.classifier = torch.nn.Linear(in_features=densenet.classifier.in_features, 
                                          out_features=out_features, 
                                          bias=True)
        
    def forward(self, x):
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1,1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
        
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        
        class_weights = self.hparams.class_weights.to(self.device)
        train_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, weight=class_weights)
        
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        preds = torch.nn.functional.softmax(outputs, dim=0)

        class_weights = self.hparams.class_weights.to(self.device)
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, weight=class_weights)
        
        val_avg_precision = self.average_precision(preds, targets)
        
        self.log_dict({ "val_loss":val_loss, 
                        "val_avg_prec":val_avg_precision }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        class_weights = self.hparams.class_weights.to(self.device)
        test_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, weight=class_weights)
        self.log("Loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        f1 = torchmetrics.F1Score(num_classes=1).to(self.device)
        f1 = f1(outputs, targets.squeeze().type(torch.uint8))
        self.log("F1 score", f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        auroc = torchmetrics.classification.BinaryAUROC(num_classes=1).to(self.device)
        auroc = auroc(outputs, targets.squeeze().type(torch.uint8))
        self.log("Area Under ROC Curve", auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        avg_prec = torchmetrics.AveragePrecision(num_classes=1).to(self.device)
        avg_prec = avg_prec(outputs, targets.squeeze().type(torch.uint8))
        self.log("Average Precision", avg_prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self): 
        params_to_optimize = []

        if self.hparams.freeze_features == 'All':
            for param in self.features.parameters():
                param.requires_grad=False
                
            params_to_optimize = self.classifier.parameters()

        elif self.hparams.freeze_features == 'None':
            for param in self.parameters():
                if not param.requires_grad:
                    param.requires_grad=True
            
            params_to_optimize = self.parameters()
            
        elif self.hparams.freeze_features == 'First3':
            for name, children in self.features.named_children():
                if name in ['conv0', 'norm0', 'relu0', 'pool0', 
                                'denseblock1', 'transition1', 'denseblock2', 
                                'transition2']:
                    for params in children.parameters():
                        params.requires_grad = False
                else:
                    params_to_optimize += children.parameters()

            params_to_optimize += self.classifier.parameters()
  
        optimizer = torch.optim.SGD(params_to_optimize, 
                                    lr=self.hparams.learning_rate, 
                                    momentum=self.hparams.momentum, 
                                    weight_decay=self.hparams.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=self.hparams.T_max,
                                                                  eta_min=self.hparams.eta_min,
                                                                  verbose=True)

        return { "optimizer": optimizer,
                 "lr_scheduler": 
                   { "scheduler": lr_scheduler,
                     "interval":"epoch",
                     "frequency": 1 }, 
                }

class ResNet18(pl.LightningModule):
    def __init__(self,
                 learning_rate:float=1e-3, 
                 momentum:float=.9, 
                 weight_decay:float=1e-4, 
                 class_weights=None, 
                 freeze_features:str='False',
                 T_max:int=50,
                 eta_min:float=5e-5,
                 input_channels:int=3, 
                 out_features:int=1):

        super().__init__()
        self.automatic_optimization = True
        self.freeze_features = freeze_features
        self.average_precision = torchmetrics.AveragePrecision(num_classes=out_features, average=None)
        # This line saves the hyper_parameters so they can be called using self.hparams...
        self.save_hyperparameters('learning_rate', 
                                  'momentum', 
                                  'weight_decay', 
                                  'class_weights',
                                  'freeze_features',
                                  'T_max',
                                  'eta_min')

        resnet = torchvision.models.resnet18(weights='DEFAULT')
        #print('original resnet18')
        #print(resnet)
        self.features = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        #self.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classifier = torch.nn.Linear(in_features=resnet.fc.in_features, 
                                          out_features=out_features, 
                                          bias=True)
        #print('New Features')
        #print(self.features)
        #print('New Classifier')
        #print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        class_weights = self.hparams.class_weights.to(self.device)
        train_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, weight=class_weights)

        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        preds = torch.sigmoid(outputs)

        class_weights = self.hparams.class_weights.to(self.device)
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, weight=class_weights)

        val_avg_precision = self.average_precision(preds, targets)

        self.log_dict({ "val_loss":val_loss, 
                        "val_avg_prec":val_avg_precision }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        class_weights = self.hparams.class_weights.to(self.device)
        test_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, weight=class_weights)

        f1 = torchmetrics.F1Score(num_classes=1).to(self.device)
        f1 = f1(outputs, targets.squeeze().type(torch.uint8))

        auroc = torchmetrics.classification.BinaryAUROC(num_classes=1).to(self.device)
        auroc = auroc(outputs, targets.squeeze().type(torch.uint8))

        avg_prec = torchmetrics.AveragePrecision(num_classes=1).to(self.device)
        avg_prec = avg_prec(outputs, targets.squeeze().type(torch.uint8))

        self.log_dict({ "Loss":test_loss, 
                        "F1 Score":f1,
                        "Area Under ROC Curve":auroc,
                        "Average Precision": avg_prec}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self): 
        params_to_optimize = []

        if self.hparams.freeze_features == 'All':
            for param in self.features.parameters():
                param.requires_grad=False

            params_to_optimize = self.classifier.parameters()

        elif self.hparams.freeze_features == 'None':
            for param in self.parameters():
                if not param.requires_grad:
                    param.requires_grad=True
            
            params_to_optimize = self.parameters()
            
        elif self.hparams.freeze_features == 'First3':
            for name, children in self.features.named_children():
                if name in ['conv0', 'norm0', 'relu0', 'pool0', 
                                'denseblock1', 'transition1', 'denseblock2', 
                                'transition2']:
                    for params in children.parameters():
                        params.requires_grad = False
                else:
                    params_to_optimize += children.parameters()

            params_to_optimize += self.classifier.parameters()
                    
        optimizer = torch.optim.SGD(params_to_optimize, 
                                    lr=self.hparams.learning_rate, 
                                    momentum=self.hparams.momentum, 
                                    weight_decay=self.hparams.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=self.hparams.T_max,
                                                                  eta_min=self.hparams.eta_min,
                                                                  verbose=True)

        return { "optimizer": optimizer,
                 "lr_scheduler": 
                   { "scheduler": lr_scheduler,
                     "interval":"epoch",
                     "frequency": 1 }, 
                }

class AlexNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, momentum=.9, weight_decay=1e-4, class_weights=None, input_channels=1, out_features=2):
        super().__init__()
        self.automatic_optimization = True
        self.save_hyperparameters('learning_rate', 'momentum', 'weight_decay', 'class_weights')
        
        alexnet = torchvision.models.alexnet(weights='DEFAULT')
        print(alexnet.eval())
        self.features = alexnet.features
        self.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False)
        self.classifier = alexnet.classifier
        self.classifier[1] = torch.nn.Linear(in_features=256, 
                                            out_features=4096, 
                                            bias=True)
        self.classifier[6] = torch.nn.Linear(in_features=4096, 
                                            out_features=out_features, 
                                            bias=True)
        print(self.eval())

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
        test_f1 = torchmetrics.F1Score(num_classes=1).to(self.device)
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