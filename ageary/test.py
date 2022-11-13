import os
import torch
from pytorch_lightning.trainer.trainer import Trainer
import argparse

parser = argparse.ArgumentParser(
                    prog = 'Model tester.',
                    description = 'This script tests a model on the x-ray dataset.',
                    epilog = 'For help append test.py with --help')

parser.add_argument('--batch_size', 
                    nargs='?', 
                    default=16, 
                    help='Enter batch size for test', 
                    type=int, 
                    required=False)

parser.add_argument('--model_dir', 
                    nargs='?', 
                    default='models', 
                    help='Directory to load model from.', 
                    required=False)

parser.add_argument('--loader_dir', 
                    nargs='?', 
                    default='dataloaders', 
                    help='Directory to load dataloader from.', 
                    required=False)


#Load test dataloader
test_loader = torch.load(os.path.join(args.loader_dir, 'test_loader.pth'))

#Load Model
model_inst = torch.load(os.path.join(args.model_dir, '.ckpt'))



trainer.test(model=model_inst,
            dataloaders=test_loader,
            ckpt_path='best',
            verbose=True)