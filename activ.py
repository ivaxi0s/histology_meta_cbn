import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
from functools import partial
import collections
import pdb

dataset = [torch.rand(3,3,224,224) for _ in range(2)]

net = tmodels.resnet18(pretrained=False)

acts = collections.defaultdict(list)

for batch in dataset:
	out = net(batch)

def save_act(name, mod, inp, out):
	acts[name].append(out.cpu())
    pdb.set_trace()


for name, m in net.named_modules():
	if type(m)==nn.Conv2d or type(m)==nn.BatchNorm2d:
		m.register_forward_hook(partial(save_act, name))
pdb.set_trace()



acts = {name: torch.cat(outputs, 0) for name, outputs in acts.items()}
pdb.set_trace()
for n,w in acts.items():
	print (n, w.size())