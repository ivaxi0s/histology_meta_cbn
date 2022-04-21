import argparse
import torch
import torch.nn as nn
import data
import networks
import numpy as np
import csv
import os
import utils
import _resnet__copy
import pdb
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('-data_path', required=True, help="Path to test HDF5 files.")
parser.add_argument('-results_path', default='results/', help='Folder to save results file in')
parser.add_argument('-saved_model', default= '/home/ivsh/scratch/projects/histology_meta_cbn/models/job_models-00007.pth', help="Path to trained model file")
parser.add_argument('-batch_size', default=128, type=int, help='Batch size')
parser.add_argument('-device', default=0, type=int, help='CUDA device')
args = parser.parse_args()
print(args)

sig = nn.Sigmoid()

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test dataset
test_dataloader = DataLoader(data.PatchCamelyon(args.data_path, mode='test'), batch_size=args.batch_size, shuffle=True)

# Create the model and load the saved model file
model = _resnet__copy.resnet18().to(device)
model.load_state_dict(torch.load(args.saved_model, map_location=torch.device('cpu')))
model.to(device)
model.eval()


def test():

    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []}
    for i,batch in enumerate(test_dataloader,1):
        # Load data
        image, label, attribute = batch
        image = image.to(device)
        label = label.to(device)
        attribute= attribute.to(device)
        zero_attributes = torch.ones(attribute.shape)
        zero_attributes = zero_attributes.to(device)
        inp_tuple = (image, zero_attributes)
        predicted = model(inp_tuple)
        predicted = predicted.squeeze().to(device)
        predicted = sig(predicted).to(device)

        # Track batch results
        m = utils.metrics(predicted, label)
        print(predicted)
        print(label)
        print(m)
        for key in m.keys():
            results[key].append(m[key])

    # Get the average over all batches
    for key in results.keys():
        results[key] = np.mean(results[key])

    w = csv.writer(open(os.path.join(args.results_path, "results.csv"), "w"))
    for key, val in results.items():
        w.writerow([key, val])


if __name__ == '__main__':
    test()
