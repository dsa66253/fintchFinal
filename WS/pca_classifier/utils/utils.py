import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
def make_dataloader(data_path, shuffle=True):
    with open(data_path, 'rb') as f:
        data_pkl = pickle.load(f)
    x = np.array(data_pkl['x'],dtype=float)
    y = np.array(data_pkl['y'],dtype=float)
    x = torch.tensor(x)
    y = torch.tensor(y)
    dataset = TensorDataset(x.float(), y.long())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=shuffle)
    return dataloader

def make_dataloader_from_np(np_x, np_y):
    x = np.array(np_x,dtype=float)
    y = np.array(np_y,dtype=float)
    x = torch.tensor(x)
    y = torch.tensor(y)
    dataset = TensorDataset(x.float(), y.long())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader

def write_file(pkl_path, data):
    print(f'writing {pkl_path} ...')
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)