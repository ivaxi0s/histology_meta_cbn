import os
import csv
import random
import tarfile
import multiprocessing as mp
import tqdm
import requests
import numpy as np
import sklearn.model_selection as skms
import torch
import torch.utils.data as td
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
import wandb
import _resnet__copy
import torch.nn as nn
from cbn import CBN
from collections import defaultdict
from functools import partial


from dataloader import DatasetBirds, pad

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = 'results'
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

class SaveAct:
    def __init__(self):
        self.layer_outputs = defaultdict(lambda x:None)
        
    def __call__(self, name, module, module_in, module_out):
        self.layer_outputs[name] = module_out

# create an output folder
os.makedirs(OUT_DIR, exist_ok=True)

# fill padded area with ImageNet's mean pixel value converted to range [0, 255]
fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
# pad images to 500 pixels
max_padding = tv.transforms.Lambda(lambda x: pad(x, fill=fill))

fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))

# transform images
transforms_train = tv.transforms.Compose([
   max_padding,
   tv.transforms.RandomOrder([
       tv.transforms.RandomCrop((225, 225)),
       tv.transforms.RandomHorizontalFlip(),
       tv.transforms.RandomVerticalFlip()
   ]),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transforms_eval = tv.transforms.Compose([
   max_padding,
   tv.transforms.CenterCrop((225, 225)),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ds_train = DatasetBirds("/home/ivsh/scratch/datasets/CUB/CUB_200_2011", transform=transforms_train, train=True)
ds_val = DatasetBirds("/home/ivsh/scratch/datasets/CUB/CUB_200_2011", transform=transforms_eval, train=True)
ds_test = DatasetBirds("/home/ivsh/scratch/datasets/CUB/CUB_200_2011", transform=transforms_eval, train=False)
# pdb.set_trace()

splits = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
idx_train, idx_val = next(splits.split(np.zeros(len(ds_train)), ds_train.targets))

train_params = {'batch_size': 2}
test_params =  {'batch_size': 64}
num_epochs = 75
num_classes = 200
# instantiate data loaders
trainset = td.DataLoader(
   dataset=ds_train,
   sampler=td.SubsetRandomSampler(idx_train),
   **train_params
)
testset = td.DataLoader(
   dataset=ds_val,
   sampler=td.SubsetRandomSampler(idx_val),
   **test_params
)
test_loader = td.DataLoader(dataset=ds_test, **test_params)
# pdb.set_trace()
model = _resnet__copy.resnet50().to(DEVICE)
# pdb.set_trace()

config_defaults = {
    'learning_rate': 0.01,
    'optimizer': 'adam',
}
wandb.init(config=config_defaults)
wandb.init(entity="ivsh", project="cub")

#To convert data from PIL to tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

root = "/home/ivsh/scratch/datasets/CUB/CUB_200_2011"
path_to_attr = os.path.join(root, 'attr.npy')
attr = []
attributes = torch.from_numpy(np.load(path_to_attr))


def get_cbn_features(activations_lst, batched_attributes):
    for i in range(len(activations_lst)):
        if "conv" in activations_lst[i][0] and activations_lst[i+1][0] == activations_lst[i][0].replace(("conv"), "bn"):
            print(i)
            # pdb.set_trace()
            # cbn_model = CBN(activations_lst[i][1].shape[1], activations_lst[i][1].shape[0])
            # _, predicted_beta, predicted_gamma = cbn_model(torch.Tensor(activations_lst[i][1]).detach().float(), batched_attributes.float())
            # st = activations_lst[i+1][0] 
            # modified_w ="model."+st.replace(".", "[", 1).replace(".", "]", 1)+".weight"
            # modified_b = "model."+st.replace(".", "[", 1).replace(".", "]", 1)+".bias"
            # fin_w = modified_w[:modified_w.find("bn")] + ".bn" + modified_w[modified_w.find("bn")+2 : ]
            # fin_b = modified_b[:modified_b.find("bn")] + ".bn" + modified_b[modified_b.find("bn")+2 : ]
            # # 

            # pdb.set_trace()
            # (activations_lst[i][1].detach().float() aa = batched_attributes.float() _out = CBN(activations_lst[i][1].shape[1], activations_lst[i][1].shape[0]) # channel x batch
            # (Pdb) fe = torch.Tensor(activations_lst[i][1]).detach().float()
            # st = activations_lst[11][0] 
            # modified ="model."+st.replace(".", "[", 1).replace(".", "]", 1)+".weight"
            # fin = modified[:modified.find("bn")] + ".bn" + modified[modified.find("bn")+2 : ]
            # 


# change to model.layer1[1].bn1.weight.shape from layer1.1.bn1 -> "model." layer1 "["1"]" bn1 ".weight"
    


def test():
    # Default values for hyper-parameters we're going to sweep over
    #Load train and test set:

    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    wandb.watch(model)
    model.to(DEVICE)
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    optimizer =  torch.optim.Adam(model.parameters(),lr=wandb.config.learning_rate)
    if wandb.config.optimizer == 'sgd':
          optimizer =  torch.optim.SGD(model.parameters(),lr=wandb.config.learning_rate)
    
    for epoch in range(num_epochs):

        closs = 0
        _correctHits=0
        _total=0
        for i,batch in enumerate(trainset,1):
            saveact = SaveAct()

            for name, m in model.named_modules(): 
                if type(m)==nn.Conv2d or type(m)==nn.BatchNorm2d:
                    m.register_forward_hook(partial(saveact, name))

            data,label = batch
            batched_attributes = attributes[label]
            
            data = data.type(torch.FloatTensor)
            label = label.type(torch.long)
            batched_attributes = batched_attributes.type(torch.FloatTensor)
            data,label = data.to(DEVICE), label.to(DEVICE)
            batched_attributes = batched_attributes.to(DEVICE)
            tu = (data, batched_attributes)

            # pdb.set_trace()

            prediction = model(tu)
            prediction = prediction.type(torch.FloatTensor).to(DEVICE)


            # activations_lst = []
            # for name, output in saveact.layer_outputs.items():  activations_lst.append([name, output])
            # get_cbn_features(activations_lst, batched_attributes)
            # pdb.set_trace()
            optimizer.zero_grad()

            loss = ce_loss(prediction,label)
            closs += loss.item()
            loss.backward()
            optimizer.step()
            # pdb.set_trace()


            _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
            _total += label.size(0)
            _correctHits += (prediction==label).sum().item()
            train_acc = (_correctHits/_total)*100
            # shift_bias(model)
            # pdb.set_trace()

            #print every 1000th time
            if i%1 == 0:
                print('[%d  %d] loss: %.4f'% (epoch+1,i+1,closs/1))
                wandb.log({"Training Loss": closs/1})
                closs = 0
            # pdb.set_trace()
        # pdb.set_trace()
        print('Train Accuracy on epoch ',epoch+1,'= ',str(train_acc))
        wandb.log({"Train Accuracy" : train_acc})


    #     correctHits=0
    #     total=0
    #     for batches in testset:
    #         data,label = batches
    #         data,label = data.to(device),label.to(device)
    #         prediction = model(data).to(device)
    #         _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
    #         total += label.size(0)
    #         correctHits += (prediction==label).sum().item()
    #         test_acc = (correctHits/total)*100
            
    #     print('Test Accuracy on epoch ',epoch+1,'= ',str(test_acc))
    #     wandb.log({"Test Accuracy" : test_acc})
        
    # correctHits=0
    # total=0
    # for logging prediction on one batch of image
    # val_images = []
    # true_labels = []
    # pred_labels = []

    # for step, batches in enumerate(testset):
    #     data,label = batches
    #     if step<1:
    #       val_images.extend(data)
    #       true_labels.extend(label)
    #     data,label = data.to(device),label.to(device)
    #     prediction = net(data)
    #     if step<1:
    #       pred_labels.extend(prediction.cpu())
    #     _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
    #     total += label.size(0)
    #     correctHits += (prediction==label).sum().item()

    # print('Accuracy = '+str((correctHits/total)*100))
    # wandb.log({"examples": [wandb.Image(np.rollaxis(image.numpy(), 0, 3)) for image in val_images]})

test()