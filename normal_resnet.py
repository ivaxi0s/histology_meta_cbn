import argparse, copy
import numpy as np
import torch
import torch.optim as optim

import networks
import data
import utils
import _resnet__copy
import torch.nn as nn
import pdb
from torch.utils.data import DataLoader
from collections import defaultdict
from functools import partial
import wandb
import statistics as st
torch.cuda.empty_cache()


torch.manual_seed(0)

class SaveAct:
    def __init__(self):
        self.layer_outputs = defaultdict(lambda x:None)
        
    def __call__(self, name, module, module_in, module_out):
        self.layer_outputs[name] = module_out

parser = argparse.ArgumentParser()
parser.add_argument('-data_path', required=True, help="Path to PCam HDF5 files.")
parser.add_argument('-save_path', default='models/', help="Path to save trained models")
parser.add_argument('-lr', default=1e-2, type=float, help="Learning rate")
parser.add_argument('-batch_size', default=128, type=int, help="Batch size")
parser.add_argument('-n_iters', default=10000, type=int, help="Number of train iterations")
parser.add_argument('-device', default=0, type=int, help="CUDA device")
parser.add_argument('-save_freq', default=1000, type=int, help="Frequency to save trained models")
parser.add_argument('-visdom_freq', default=250, type=int, help="Frequency  plot training results")
args = parser.parse_args()
print(args)
num_epochs = 3
# Dataset

train_dataloader = DataLoader(data.PatchCamelyon(args.data_path, mode='train', augment=True), batch_size=args.batch_size, shuffle=True)
valid_dataloader = DataLoader(data.PatchCamelyon(args.data_path, mode='valid'), batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(data.PatchCamelyon(args.data_path, mode='test'), batch_size=args.batch_size, shuffle=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
model = networks.CamelyonClassifier().to(device)
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8)

# Loss function
criterion = nn.BCELoss()

# Visdom writer
# writer = utils.Writer()


sig = nn.Sigmoid()
mse = nn.MSELoss()

def train():
    model.train()

    losses = []
    train_acc = []    

    for epoch in range(num_epochs):
        losses = []
        _correctHits = 0
        _total = 0
        losses2 = []
        for i,batch in enumerate(train_dataloader,1):

            # Zero gradient
            optimizer.zero_grad()

            # Load data to GPU
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            predicted = model(image)
            predicted = predicted.squeeze().to(device)
            predicted = sig(predicted).to(device)

            # Loss
            loss = criterion(predicted, label)
            losses.append(loss.data.item())
            loss.backward()
            optimizer.step()



            metrics = utils.metrics(predicted, label)
            train_acc.append(metrics['accuracy'] ) 

            print("\n Iteration: {:06d} of {:06d}\t\t Train CBN Loss: {:.4f} \t\t ".format(epoch, i, np.mean(losses)), 
                end="\n\n")

        train_acc_lst = []
        train_acc_lst.append(st.mean(train_acc))
        lst = np.asarray(losses)
        np.savetxt("losses.csv", lst, delimiter=",")
        val_acc_lst = []
        v = validation()
        val_acc_lst.append(v)
        test_acc_lst = []
        t = test()
        test_acc_lst.append(t)
        print("accuracies:" , st.mean(train_acc), v, t)
        torch.save(model.state_dict(), 'models_renet/model-{:05d}.pth'.format(epoch))

        # lst.append(torch.tensor(losses).mean(), torch.tensor(accuracy).mean(), torch.tensor(f1).mean(), \
        #    torch.tensor(specificity).mean(), torch.tensor(precision).mean())
    lst = np.asarray(train_acc_lst)
    np.savetxt("train_acc_reset.csv", lst, delimiter=",")
    lst = np.asarray(val_acc_lst)
    np.savetxt("val_acc_resnet.csv", lst, delimiter=",")
    lst = np.asarray(test_acc_lst)
    np.savetxt("test_acc_resnet.csv", lst, delimiter=",")    
        
def validation():
    model.eval()

    losses = []
    accuracy = []
    f1 = []
    specificity = []
    precision = []

    for i,batch in enumerate(valid_dataloader,1):

        # Zero gradient
        optimizer.zero_grad()

        # Load data to GPU
        image, label = batch
        image = image.to(device)
        label = label.to(device)

        predicted = model(image).squeeze()

        # Loss
        loss = criterion(predicted, label)
        losses.append(loss.data.item())

        

        # Metrics
        metrics = utils.metrics(predicted, label)
        accuracy.append(metrics['accuracy'])
        f1.append(metrics['f1'])
        specificity.append(metrics['specificity'])
        precision.append(metrics['precision'])

    return torch.tensor(accuracy).mean()

def test():
    model.eval()

    losses = []
    accuracy = []
    f1 = []
    specificity = []
    precision = []

    for i,batch in enumerate(test_dataloader,1):

        # Zero gradient
        optimizer.zero_grad()

        # Load data to GPU
        image, label = batch
        image = image.to(device)
        label = label.to(device)

        predicted = model(image).squeeze()

        # Loss
        loss = criterion(predicted, label)
        losses.append(loss.data.item())

        

        # Metrics
        metrics = utils.metrics(predicted, label)
        accuracy.append(metrics['accuracy'])
        f1.append(metrics['f1'])
        specificity.append(metrics['specificity'])
        precision.append(metrics['precision'])

    return torch.tensor(accuracy).mean()


if __name__ == '__main__':
    train()
