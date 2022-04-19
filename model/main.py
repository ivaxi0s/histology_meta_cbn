import resnet
import torch, pdb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

attr = torch.rand(312)
m = resnet.resnet50(4).to(DEVICE)
pdb.set_trace()
xx = torch.rand(4,3,224,224)
att = torch.rand(312)
abc = m(xx, att)
pdb.set_trace()