import os, pdb
import h5py
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import csv


class PatchCamelyon(data_utils.Dataset):

    def __init__(self, path, mode='train', n_iters=None, augment=False):
        super().__init__()

        self.n_iters = n_iters

        assert mode in ['train', 'valid', 'test']
        base_name = "camelyonpatch_level_2_split_{}_{}.h5"
        csv_base_name = "camelyonpatch_level_2_split_{}_meta.csv"

        print('\n')
        print("# " * 50)
        print('Loading {} dataset...'.format(mode))

        # Open the files
        h5X = h5py.File(os.path.join(path, base_name.format(mode, 'x')), 'r')
        h5y = h5py.File(os.path.join(path, base_name.format(mode, 'y')), 'r')
        file = open(os.path.join(path, csv_base_name.format(mode)))
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)
        rows = []
        for row in csvreader: rows.append(row)
        meta_all = np.array(rows)
        meta = meta_all[:, [1,2]]
        meta_x = meta[:,0].astype(np.float)
        res_a = (meta_x - meta_x.mean())/meta_x.std()
        meta_y = meta[:,1].astype(np.float)
        res_b = (meta_y - meta_y.mean())/meta_y.std()
        attr = np.vstack((res_a, res_b))
        attr = np.swapaxes(attr,0,1)

        # Read into numpy array
        self.X = np.array(h5X.get('x'))
        self.y = np.array(h5y.get('y'))
        self.attr = attr

        print('Loaded {} dataset with {} samples'.format(mode, len(self.X)))
        print("# " * 50)

        if augment:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ColorJitter(brightness=.5, saturation=.25, hue=.1, contrast=.5),
                                                 transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor()])

    def __getitem__(self, item):
        idx = item % self.__len__()
        _slice = slice(idx, (idx + 1))
        images = self._transform(self.X[_slice])
        images = torch.squeeze(images, 0)
        labels = torch.tensor(self.y[_slice].astype(np.float32)).view(-1, 1)
        labels = labels.squeeze()
        attributes = self.attr[_slice]
        attributes = attributes.squeeze()
        if labels == 0 : attributes = np.zeros(attributes.shape)
        attributes = np.float32(attributes)

        return images, labels, attributes

    def _transform(self, images):
        tensors = []
        for image in images:
            tensors.append(self.transform(image))
        return torch.stack(tensors)

    def __len__(self):
        return len(self.X) 
