import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import os

label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

def create_data_loader(data_dir, label_df, load_target=True, batch_size=128, shuffle=False, num_workers=16, idx=None):
    if idx is not None and len(idx) > 0:
        label_df = label_df.iloc[idx]

    if load_target:
        onehot = [target2onehot(t) for t in label_df.Target]
        tgt = dict(zip(label_df.Id, np.array(onehot)))
    else:
        tgt = dict(zip(label_df.Id, np.zeros(len(label_df))))

    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': num_workers}

    ds = HmDataset(label_df.Id.tolist(), tgt, data_dir)
    return DataLoader(ds, **params)

def target2onehot(target):
    y = np.zeros(len(label_names))
    t = [int(t) for t in target.split(' ')]
    y[t] = 1
    return y 


class HmDataset(Dataset):
    def __init__(self, list_IDs, labels, data_dir):
        'Initialization'
        self.labels = labels # One hot labels
        self.list_IDs = list_IDs
        self.data_dir = data_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        y = self.labels[ID]
        
        # TODO: fix that
        X = np.empty((4, 512, 512))
        X[0,:,:] = cv2.imread(os.path.join(self.data_dir, ID + "_green" + ".png"), 0)
        X[1,:,:] = cv2.imread(os.path.join(self.data_dir, ID + "_red" + ".png"), 0)
        X[2,:,:] = cv2.imread(os.path.join(self.data_dir, ID + "_blue" + ".png"), 0)
        X[3,:,:] = cv2.imread(os.path.join(self.data_dir, ID + "_yellow" + ".png"), 0)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)