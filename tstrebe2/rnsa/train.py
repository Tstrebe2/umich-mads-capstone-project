import argparse

import torchvision
import torch
import pydicom as dicom

import numpy as np
import pandas as pd
from functools import partial
import rnsa

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description='This module fine-tunes a pre-trained densenet model on 2018 RNSA Pneumonia Detection challenge data')

parser.add_argument('--num_frozen_epochs', nargs='?', default=0, help='Enter the # of epochs to train while feature hidden layers are frozen.', type=int, required=False)
parser.add_argument('--num_fine_tune_epochs', nargs='?', default=10, help='Enter the # of epochs for fine tuning.', type=int, required=False)
parser.add_argument('--restore_checkpoint', nargs='?', default=1, choices=[0, 1], help='0 or 1 boolean to restore model from last checkpoint.', type=int, required=False)
parser.add_argument('--frozen_batch_size', nargs='?', default=32, help='Batch size for training while feature hidden layers are frozen.', type=int, required=False)
parser.add_argument('--ft_batch_size', nargs='?', default=22, help='Batch size for fine tuning.', type=int, required=False)
args = parser.parse_args()

NUM_FROZEN_EPOCHS = args.num_frozen_epochs
NUM_FT_EPOCHS = args.num_fine_tune_epochs
RESTORE_CHECKPOINT = bool(args.restore_checkpoint)
FROZEN_BATCH_SIZE = args.frozen_batch_size
FT_BATCH_SIZE = args.ft_batch_size

SKIP_FROZEN = NUM_FROZEN_EPOCHS <= 0

print('Num frozen epochs:', NUM_FROZEN_EPOCHS)
print('Num fine-tune epochs:', NUM_FT_EPOCHS)
print('Restore checkpoint?:', RESTORE_CHECKPOINT)
print('Frozen batch size:', FROZEN_BATCH_SIZE)
print('Fine-tune batch size:', FT_BATCH_SIZE)
print('Skip frozen layer runing:', SKIP_FROZEN)

model_save_path = '/home/tstrebel/models/rnsa-densenet.pt'
train_img_dir = '/home/tstrebel/assets/rnsa-pneumonia/train-images'
annotations_file_path = '/home/tstrebel/assets/rnsa-pneumonia/stage_2_train_labels.csv.zip'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_df = pd.read_csv(annotations_file_path).groupby('patientId').first().reset_index()

X_train, X_test = train_test_split(label_df, test_size=.2, stratify=label_df.Target, random_state=99)
X_val, X_test, = train_test_split(X_test, test_size=.4, stratify=X_test.Target, random_state=99)
train_ix, val_ix, test_ix = X_train.index.tolist(), X_val.index.tolist(), (X_test.index.tolist())
del(X_train)
del(X_val)
del(X_test)
print('train {:,} - validate {:,} - test {:,}'.format(len(train_ix), len(val_ix), len(test_ix)))
# Single channel mean & standard deviation.
mean = [0.5]
std = [0.225]

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation((-2, 2)),
    torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

label_transform = torchvision.transforms.Compose([
    partial(torch.tensor, dtype=torch.float),
    partial(torch.unsqueeze, dim=0),
])
    
train_dataset = rnsa.RNSADataset(train_img_dir, annotations_file_path, train_ix, train_transform, label_transform)
val_dataset = rnsa.RNSADataset(train_img_dir, annotations_file_path, val_ix, val_transform, label_transform)

if not SKIP_FROZEN:
    if RESTORE_CHECKPOINT:
        model, best_loss, best_acc, lr = rnsa.load_checkpoint(model_save_path, 'cpu')
        print('checkpoint best loss: {:.4}'.format(best_loss))
        print('checkpoint best acc: {:.4}'.format(best_acc))
    else:
        model = rnsa.Densenet121(torchvision.models.densenet121(weights='DEFAULT'))
        best_loss = np.inf
        best_acc = 0.0
        lr = 1e-3

    model = model.to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr, momentum=.9, weight_decay=1e-4)
    lr_scheduler = rnsa.LRScheduler(optimizer)

criterion = torch.nn.BCEWithLogitsLoss()

if not SKIP_FROZEN:
    rnsa.train_model(model,  
                     model_save_path,
                     train_dataset, 
                     val_dataset,
                     optimizer, 
                     criterion, 
                     device, 
                     batch_size=FROZEN_BATCH_SIZE,
                     num_epochs=NUM_FROZEN_EPOCHS,
                     init_best_loss=best_loss,
                     init_best_acc=best_acc)
    
model, best_loss, best_acc, lr = rnsa.load_checkpoint(model_save_path, 'cpu')
model = model.to(device)

for param in model.features.parameters():
    if not param.requires_grad:
        param.requires_grad = True

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=.9, weight_decay=1e-4)
lr_scheduler = rnsa.LRScheduler(optimizer)

print('checkpoint best loss: {:.4}'.format(best_loss))
print('checkpoint best acc: {:.4}'.format(best_acc))

rnsa.train_model(model,  
                 model_save_path,
                 train_dataset, 
                 val_dataset,
                 optimizer, 
                 criterion, 
                 device,
                 batch_size=FT_BATCH_SIZE,
                 num_epochs=NUM_FT_EPOCHS,
                 init_best_loss=best_loss,
                 init_best_acc=best_acc)