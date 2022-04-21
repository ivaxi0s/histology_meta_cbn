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

torch.manual_seed(10)



class SaveAct:
    def __init__(self):
        self.layer_outputs = defaultdict(lambda x:None)
        
    def __call__(self, name, module, module_in, module_out):
        self.layer_outputs[name] = module_out

parser = argparse.ArgumentParser()
parser.add_argument('-data_path', required=True, help="Path to PCam HDF5 files.")
parser.add_argument('-save_path', default='models/', help="Path to save trained models")
parser.add_argument('-lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('-batch_size', default=128, type=int, help="Batch size")
parser.add_argument('-n_iters', default=10000, type=int, help="Number of train iterations")
parser.add_argument('-device', default=0, type=int, help="CUDA device")
parser.add_argument('-save_freq', default=1000, type=int, help="Frequency to save trained models")
parser.add_argument('-visdom_freq', default=250, type=int, help="Frequency  plot training results")
args = parser.parse_args()
print(args)
num_epochs = 20
# Dataset

train_dataloader = DataLoader(data.PatchCamelyon(args.data_path, mode='train', augment=True), batch_size=args.batch_size, shuffle=True)
valid_dataloader = DataLoader(data.PatchCamelyon(args.data_path, mode='valid'), batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(data.PatchCamelyon(args.data_path, mode='test'), batch_size=args.batch_size, shuffle=True)


# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
model = _resnet__copy.resnet18().to(device)
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8)

# Loss function
criterion = nn.BCELoss()

# Visdom writer
# writer = utils.Writer()

def get_para(model):
    save_all_layers = []
    for name, layer in model.named_modules():
        save_all_layers.append([name, layer])
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            if 'layer' in name:
                shape_0 = layer.weight.data.shape[0] 
                shape_1 = layer.weight.data.shape[1]
                layer.weight.data = torch.ones(shape_0, shape_1).to(device)
                layer.bias.data = torch.zeros(shape_0).to(device)
            # if 'layer' in name:
            #     layer.weight = torch.ones(2)

    return model.to(device)


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children



sig = nn.Sigmoid()
mse = nn.MSELoss()

def train():
    model.train()
    train_acc = []
    test_acc_lst = []
    train_acc_lst = []
    val_acc_lst = []    

    for epoch in range(num_epochs):
        losses = []
        _correctHits = 0
        _total = 0
        losses2 = []
        
        for i,batch in enumerate(train_dataloader,1):

            # Zero gradient
            optimizer.zero_grad()

            # Load data to GPU
            image, label, attribute = batch
            image = image.to(device)
            label = label.to(device)
            attribute = attribute.to(device)
            inp_tuple = (image, attribute)
            predicted = model(inp_tuple)
            predicted = predicted.squeeze().to(device)
            predicted = sig(predicted).to(device)


            no_cbn_model_1 =get_para(model)
            no_cbn_model_1 = no_cbn_model_1.to(device)
            zero_attributes = torch.ones(attribute.shape)
            zero_attributes = zero_attributes.to(device)
            inp_tuple2 = (image, zero_attributes)
            # pdb.set_trace()
            predicted2 = no_cbn_model_1(inp_tuple2)
            predicted2 = predicted2.squeeze().to(device) 
            predicted2 = sig(predicted2).to(device)

            # Loss
            loss = criterion(predicted, label)
            loss2 = mse(predicted, predicted2)
            losses2.append(loss2.data.item())
            losses.append(loss.data.item())

            loss = loss + loss2

            # Back-propagation
            loss.backward()
            optimizer.step()

            metrics = utils.metrics(predicted, label) 
            train_acc.append(metrics['accuracy'] )         
            print("\n Iteration: {:06d} of {:06d}\t\t Train CBN Loss: {:.4f} \t\t MSE  {:.4f} \t\t".format(epoch, i, np.mean(losses), np.mean(losses2)), 
                end="\n\n")
        
        
        train_acc_lst.append(st.mean(train_acc))
        lst = np.asarray(losses)
        np.savetxt("losses.csv", lst, delimiter=",")
        lst = np.asarray(losses2)
        np.savetxt("losses2.csv", lst, delimiter=",")
        torch.save(model.state_dict(), 'models/job_models-{:05d}.pth'.format(epoch))
        v = validation()
        val_acc_lst.append(v)
        
        t = test()
        test_acc_lst.append(t)
        print("accuracies:" , st.mean(train_acc), v, t)


    lst = np.asarray(train_acc_lst)
    np.savetxt("train_acc_cbn_tn.csv", lst, delimiter=",")
    lst = np.asarray(val_acc_lst)
    np.savetxt("val_acc_cbn_tn.csv", lst, delimiter=",")
    lst = np.asarray(test_acc_lst)
    np.savetxt("test_acc_cbn_Tn.csv", lst, delimiter=",")


def validation():
    model.eval()

    losses = []
    accuracy = []
    f1 = []
    specificity = []
    precision = []

    for i,batch in enumerate(valid_dataloader,1):

        # Load data to GPU
        image, label, attribute = batch
        image = image.to(device)
        label = label.to(device)
        zero_attributes = torch.ones(attribute.shape)
        zero_attributes = zero_attributes.to(device)
        inp_tuple = (image, zero_attributes)
        predicted = model(inp_tuple)
        predicted = predicted.squeeze().to(device)
        predicted = sig(predicted).to(device)


        loss = criterion(predicted, label)
        metrics = utils.metrics(predicted, label)
        
        # Metrics

        metrics = utils.metrics(predicted, label)
        accuracy.append(metrics['accuracy'])
        f1.append(metrics['f1'])
        specificity.append(metrics['specificity'])
        precision.append(metrics['precision'])

        print("acc_val:", metrics['accuracy'] )

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
        image, label, attribute = batch
        image = image.to(device)
        label = label.to(device)
        zero_attributes = torch.ones(attribute.shape)
        zero_attributes = zero_attributes.to(device)
        inp_tuple = (image, zero_attributes)
        predicted = model(inp_tuple)
        predicted = predicted.squeeze().to(device)
        predicted = sig(predicted).to(device)


        loss = criterion(predicted, label)
        metrics = utils.metrics(predicted, label)
        
        # Metrics

        metrics = utils.metrics(predicted, label)
        accuracy.append(metrics['accuracy'])
        f1.append(metrics['f1'])
        specificity.append(metrics['specificity'])
        precision.append(metrics['precision'])

    return torch.tensor(accuracy).mean()


if __name__ == '__main__':
    train()
