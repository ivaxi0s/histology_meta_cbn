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




class SaveAct:
    def __init__(self):
        self.layer_outputs = defaultdict(lambda x:None)
        
    def __call__(self, name, module, module_in, module_out):
        self.layer_outputs[name] = module_out

parser = argparse.ArgumentParser()
parser.add_argument('-data_path', required=True, help="Path to PCam HDF5 files.")
parser.add_argument('-save_path', default='models/', help="Path to save trained models")
parser.add_argument('-lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('-batch_size', default=2, type=int, help="Batch size")
parser.add_argument('-n_iters', default=10000, type=int, help="Number of train iterations")
parser.add_argument('-device', default=0, type=int, help="CUDA device")
parser.add_argument('-save_freq', default=1000, type=int, help="Frequency to save trained models")
parser.add_argument('-visdom_freq', default=250, type=int, help="Frequency  plot training results")
args = parser.parse_args()
print(args)
num_epochs = 2
# Dataset

train_dataloader = DataLoader(data.PatchCamelyon(args.data_path, mode='train', augment=True), batch_size=args.batch_size, shuffle=True)
valid_dataloader = DataLoader(data.PatchCamelyon(args.data_path, mode='valid'), batch_size=args.batch_size, shuffle=True)

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
                layer.weight.data = torch.ones(shape_0, shape_1)
                layer.bias.data = torch.zeros(shape_0)
            # if 'layer' in name:
            #     layer.weight = torch.ones(2)

    return model


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

def train():
    model.train()

    losses = []
    

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
            predicted = sig(predicted)


            no_cbn_model_1 =get_para(model)
            zero_attributes = torch.zeros(attribute.shape)
            inp_tuple2 = (image, zero_attributes)
            predicted2 = no_cbn_model_1(inp_tuple2)
            predicted2 = predicted2.squeeze().to(device) 
            predicted2 = sig(predicted2)

            # Loss
            loss = criterion(predicted, label)
            loss2 = criterion(predicted2, label)
            losses2.append(loss2.data.item())

            losses.append(loss.data.item())

            # Back-propagation
            loss.backward()
            optimizer.step()
            _correctHits += (predicted==label).sum().item()
            _total += label.size(0)
            pdb.set_trace()

            metrics = utils.metrics(predicted, label)

            print("Iteration: {:06d} of {:06d}\t\t Train Loss: {:.4f}".format(epoch, i, np.mean(losses)))

            print("Iteration: {:06d} of {:06d}\t\t Train Loss2: {:.4f}".format(epoch, i, np.mean(losses2)))

            train_acc = (_correctHits/_total)*100

        print('Train Accuracy on epoch ',epoch+1,'= ',str(train_acc))

            # if idx % args.visdom_freq == 0:

            #     # Get loss and metrics from validation set
            # val_loss, accuracy, f1, specificity, precision = validation()

            #     # Plot train and validation loss
            #     writer.plot('loss', 'train', 'Loss', idx, np.mean(losses))
            #     writer.plot('loss', 'validation', 'Loss', idx, val_loss)

            #     # Plot metrics
            #     writer.plot('accuracy', 'test', 'Accuracy', idx, accuracy)
            #     writer.plot('specificity', 'test', 'Specificity', idx, specificity)
            #     writer.plot('f1', 'test', 'F1', idx, f1)
            #     writer.plot('precision', 'test', 'Precision', idx, precision)

            #     # Print output
            # print("\nIteration: {:04d} of {:04d}\t\t Valid Loss: {:.4f}".format(idx, args.n_iters, val_loss),
            #         end="\n\n")

                # Set model to training mode again
                # model.train()

            # if idx % args.save_freq == 0:
            #     torch.save(model.state_dict(), 'models/model-{:05d}.pth'.format(idx))


# def validation():
#     pdb.set_trace()
#     model.eval()

#     losses = []
#     accuracy = []
#     f1 = []
#     specificity = []
#     precision = []

    # for epoch in range(num_epochs):
    #     losses = []
    #     for i,batch in enumerate(train_dataloader,1):
    #         pdb.set_trace()

    #         # Zero gradient
    #         optimizer.zero_grad()

    #         # Load data to GPU
    #         image, label = batch
    #         image = image.to(device)
    #         label = label.to(device)

    #         predicted = model(image)

    #         # Loss
    #         loss = criterion(predicted, label)
    #         losses.append(loss.data.item())

            

    #         # Metrics
    #         metrics = utils.metrics(predicted, labels)
    #         accuracy.append(metrics['accuracy'])
    #         f1.append(metrics['f1'])
    #         specificity.append(metrics['specificity'])
    #         precision.append(metrics['precision'])

    # return torch.tensor(losses).mean(), torch.tensor(accuracy).mean(), torch.tensor(f1).mean(), \
    #        torch.tensor(specificity).mean(), torch.tensor(precision).mean()


if __name__ == '__main__':
    train()
