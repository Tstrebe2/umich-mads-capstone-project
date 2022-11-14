import os
import numpy as np
import cx14
from cx14_target_data import get_training_data_target_dict
import my_args
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from sklearn.utils.class_weight import compute_class_weight

def main():
    parser = my_args.get_argparser()

    args = parser.parse_args()

    if args.restore_ckpt_path == 'None':
        restore_ckpt_path=None
    else:
        restore_ckpt_path = args.restore_ckpt_path
        
    # Get data frames with file names & targets
    training_data_target_dict = get_training_data_target_dict(target_dir=args.target_dir)
    df_train = training_data_target_dict['df_train']
    df_val = training_data_target_dict['df_val']
    del(training_data_target_dict)
    
    # Create class weights to balance cross-entropy loss function
    sorted_targets = np.sort(df_train.target.values)
    class_weights = sorted_targets.shape[0] / ((np.unique(sorted_targets).shape[0] * np.bincount(sorted_targets)))
    del(sorted_targets)
    # We're going to compute standard balanced weights and then shrink them by computing the
    # standard deviation and adding 1 to avoid extreme class weights.
    class_weights = (class_weights - class_weights.mean()) / class_weights.std()
    class_weights = torch.from_numpy(class_weights + 1.0).float()
    
    # Using mean and standard deviation of 15,000 image samples from the training set.
    mean = [0.5341]
    std = [0.2232]
    
    # Define our transformations
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation((-3, 3)),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
        torchvision.transforms.Resize(512),
        torchvision.transforms.CenterCrop(448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(512),
        torchvision.transforms.CenterCrop(448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    
    # Define our datsets
    train_dataset = cx14.CX14Dataset(args.image_dir, 
                                     df_train, 
                                     transform=train_transform)
    train_loader = cx14.get_data_loader(train_dataset, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers_per_node, 
                                        shuffle=True)

    val_dataset = cx14.CX14Dataset(args.image_dir, 
                                   df_val, 
                                   transform=val_transform)
    val_loader = cx14.get_data_loader(val_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers_per_node)
    
    model_args = dict(
        learning_rate=args.init_learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        freeze_features=args.freeze_features,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
    )
    
    if restore_ckpt_path:
        # If restoring checkpoint, we'll load using hyper-params defined by
        # argparse.
        model = cx14.Densenet121.load_from_checkpoint(restore_ckpt_path, **model_args)
    else:
        # If not, we'll definte a new model that automatically loads pre-trained imagenet weights.
        model = cx14.Densenet121(**model_args)

    # Declare callbacks
    callbacks = []

    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=args.models_dir,
                                                  filename='cx14-densenet',
                                                  monitor='val_loss',
                                                  save_top_k=2,
                                                  save_last=True,
                                                  verbose=True,
                                                  mode='min',
                                                  save_weights_only=False))
    callbacks.append(pl.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=1e-4,
                                                patience=args.num_stop_rounds,
                                                verbose=True,
                                                mode='min',
                                                check_finite=True))
    callbacks.append(pl.callbacks.LearningRateMonitor())

    # Set training parameters
    trainer = Trainer(accelerator='gpu', 
                      devices=-1,
                      num_nodes=args.num_nodes,
                      callbacks=callbacks,
                      logger=True,
                      max_epochs=args.epochs,
                      # Evaluating a string is the result of a pytorch lightning bug
                      # that sets all boolean lightning module attributes to true on DDP.
                      fast_dev_run=args.fast_dev_run == 'True') # Flag to set traininer in debug mode

    # Begin fitting model
    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)
    
if __name__ == '__main__':
    main()