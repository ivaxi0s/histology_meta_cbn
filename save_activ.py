import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tmodels
from functools import partial
import collections
import pdb


class SaveAct:
    def __init__(self):
        self.layer_outputs = collections.defaultdict(lambda x:None)
        
    def __call__(self, name, module, module_in, module_out):
        self.layer_outputs[name] = module_out


def get_bn_conv_layers(net, dataset):
    saveact = SaveAct()
    for name, m in net.named_modules(): 
        if type(m)==nn.Conv2d or type(m)==nn.BatchNorm2d:
            m.register_forward_hook(partial(saveact, name))
    out = net(dataset) #need to create output register forward hook
    save_activations = []
    for name, output in saveact.layer_outputs.items():
        save_activations.append([name, output])
        print(name, output.shape)
    return saveact.layer_outputs
pdb.set_trace()
dataset = torch.rand(3,3,224,224).cpu()
net = tmodels.resnet18(pretrained=False).cpu() 

op = get_bn_conv_layers(net, dataset)
pdb.set_trace()