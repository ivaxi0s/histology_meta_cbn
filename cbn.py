import torch
import torch.nn as nn
import pdb
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# default `log_dir` is "runs" - we'll be more specific here

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CBN(nn.Module):
    def __init__(self, channel):
        super(CBN, self).__init__()
        self.channel = channel
        # self.biases = biases
        # self.attributes = attributes
        # self.batch_size = batch_size
        self.betas = nn.Parameter(torch.zeros(128, self.channel))
        self.gammas = nn.Parameter(torch.ones(128, self.channel))
        # self.height = height
        # self.width = width
        
        self.fc_gamma = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.channel),
            ).to(device)
        self.fc_beta = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.channel),
            ).to(device)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    def create_cbn(self, attributes):
        delta_betas = self.fc_beta(attributes)
        delta_gammas = self.fc_gamma(attributes)

        return delta_betas, delta_gammas

    def forward(self, feature, attributes):
        batch_size, channel, height, width = feature.data.shape
        if attributes.mean() != 1 : 
            delta_betas, delta_gammas = self.create_cbn(attributes)
        

        if attributes.mean() == 1 : 
            delta_betas, delta_gammas = (torch.zeros(channel) , torch.ones(channel))
            delta_betas = delta_betas.to(device)
            delta_gammas = delta_gammas.to(device)
            # pdb.set_trace()

        
        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas
        
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = torch.stack([betas_cloned]*height, dim=2)
        betas_expanded = torch.stack([betas_expanded]*width, dim=3)

        gammas_expanded = torch.stack([gammas_cloned]*height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded]*width, dim=3)

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+1.0e-5)

        # print("Means of weights & biases:", gammas_expanded.mean(), betas_expanded.mean())

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        # print("Before and After MLP:", feature_normalized, out)
        return out

# pdb.set_trace()
# inp = torch.randn(4,312) # batchxattr
# fea = torch.randn(4, 64, 94, 94) # prev conv layer
# out = CBN(64, 4) # channel x batch
# abc = out(fea, inp)
