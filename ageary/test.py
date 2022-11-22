import os
import torch
from pytorch_lightning.trainer.trainer import Trainer
import my_args
import pytorch_lightning as pl
import models
import data
import gc

def main():
    parser = my_args.get_argparser()
    args = parser.parse_args()

    #Load test dataloader
    test_loader = torch.load(os.path.join(args.loader_dir, args.loader))

    # Get data frames with file names & targets
    training_data_target_dict = data.get_training_data_target_dict(args.targets_path)
    df_test = training_data_target_dict['df_test']
    del training_data_target_dict
    
    # clean up memory
    gc.collect()
    
    # Create class weights to balance cross-entropy loss function
    class_weights = torch.tensor([1.5], dtype=torch.float)
    
    # Get datasets & loaders
    test_dataset = data.get_dataset(args.image_dir, df_test)
    test_loader = data.get_data_loader(test_dataset, 
                                           batch_size=args.batch_size, 
                                           num_workers=args.num_workers_per_node)

    #Load Model
    if args.model.lower() == 'densenet':
        model_inst = models.DenseNet121.load_from_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint))
    elif args.model.lower() == 'resnet':
        model_inst = models.ResNet.load_from_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint))
    elif args.model.lower() == 'alexnet':
        print('hello')
        model_inst = models.AlexNet.load_from_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint))

    trainer = Trainer(accelerator='gpu',
                    devices=-1,
                    num_nodes=1,
                    logger=True)

    trainer.test(model=model_inst,
                dataloaders=test_loader,
                verbose=True)

if __name__ == '__main__':
    main()