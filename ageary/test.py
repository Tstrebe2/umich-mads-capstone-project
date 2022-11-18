import os
import torch
from pytorch_lightning.trainer.trainer import Trainer
import argparse
import pytorch_lightning as pl
import model

parser = argparse.ArgumentParser(
                    prog = 'Model tester.',
                    description = 'This script tests a model on the x-ray dataset.',
                    epilog = 'For help append test.py with --help')

parser.add_argument('--model',
                    nargs='?', 
                    default='densenet121', 
                    help='Can be densenet121 densenet201 or alexnet', 
                    required=False)

parser.add_argument('--loader_dir', 
                    nargs='?', 
                    default='dataloaders', 
                    help='Directory to load dataloader from.', 
                    required=False)

parser.add_argument('--loader',
                    nargs='?', 
                    default='test_loader.pth', 
                    help='Dataloader file name', 
                    required=False)

parser.add_argument('--checkpoint_dir', 
                    nargs='?', 
                    default='models', 
                    help='Directory to load model from.', 
                    required=False)

parser.add_argument('--checkpoint',
                    nargs='?', 
                    default='model.ckpt', 
                    help='Model Checkpoint file name', 
                    required=False)



args = parser.parse_args()

#Load test dataloader
test_loader = torch.load(os.path.join(args.loader_dir, args.loader))

#Load Model
if args.model.lower() == 'densenet121':
    model_inst = model.DenseNet121.load_from_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint))
elif args.model.lower() == 'densenet201':
    model_inst = model.DenseNet201.load_from_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint))
elif args.model.lower() == 'alexnet':
    print('hello')
    model_inst = model.AlexNet.load_from_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint))

trainer = Trainer(accelerator='gpu',
                  devices=-1,
                  num_nodes=1,
                  logger=True)

trainer.test(model=model_inst,
            dataloaders=test_loader,
            verbose=True)