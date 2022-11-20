import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer

import torchvision
import torch

import gc

import rnsa
import rnsa_data
import my_args

def main():
    parser = my_args.get_argparser()

    args = parser.parse_args()

    restore_ckpt_path = None if args.restore_ckpt_path == 'None' else args.restore_ckpt_path
        
    # Get data frames with file names & targets
    training_data_target_dict = rnsa_data.get_training_data_target_dict(args.targets_path)
    df_train = training_data_target_dict['df_train']
    df_val = training_data_target_dict['df_val']
    del training_data_target_dict
    
    # clean up memory
    gc.collect()
    
    # Create class weights to balance cross-entropy loss function
    class_weights = torch.tensor([1.5], dtype=torch.float)
    
    # Get datasets & loaders
    train_dataset = rnsa_data.get_dataset(args.image_dir, df_train, train=True)
    train_loader = rnsa_data.get_data_loader(train_dataset, 
                                             batch_size=args.batch_size, 
                                             num_workers=args.num_workers_per_node, 
                                             shuffle=True)

    val_dataset = rnsa_data.get_dataset(args.image_dir, df_val)
    val_loader = rnsa_data.get_data_loader(val_dataset, 
                                           batch_size=args.batch_size, 
                                           num_workers=args.num_workers_per_node)
    
    model_args = dict(
        learning_rate=args.init_learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        freeze_features=args.freeze_features,
        T_max=args.epochs,
        eta_min=args.eta_min,
    )
    
    if restore_ckpt_path:
        # If restoring checkpoint, we'll load using hyper-params defined by
        # argparse.
        model = rnsa.Densenet121.load_from_checkpoint(restore_ckpt_path, **model_args)
    else:
        # If not, we'll definte a new model that automatically loads pre-trained imagenet weights.
        model = rnsa.Densenet121(**model_args)
        
    # Declare callbacks
    callbacks = []

    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=args.models_dir,
                                                  filename='rnsa-densenet',
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