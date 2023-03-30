import torch
from torch_geometric.data import Dataset, DataLoader
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang


class MyDataset(Dataset):
    def __init__(self, root='dataset/', file_name=None):
        super().__init__()
        self.root = root
        self.fn = np.load(file_name)

    def len(self):
        return len(self.fn)

    def get(self, idx):
        fn = self.fn[idx]
        data = torch.load(fn)
        return data


class DatasetLDS(Dataset):
    def __init__(self, root='dataset/', file_name=None, weights='weights.npy', y_min=-20.0, y_max=20.0):
        super().__init__()
        self.root = root
        self.fn = np.load(file_name)
        self.weights = np.load(weights)
        self.y_min = y_min
        self.y_max = y_max
        self.dh = (y_max - y_min) / 500

    def len(self):
        return len(self.fn)

    def get(self, idx):
        fn = self.fn[idx]
        data = torch.load(fn)
        index = torch.div(data.y-self.y_min, self.dh, rounding_mode='floor')
        index = np.asarray(index, dtype=int)
        weights = self.weights[index]
        data.weights = torch.tensor(weights)
        return data


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def lds_weights(targets, reweight_sqrt=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    target_hist, bins = np.histogram(targets.flatten(), bins=500, range=(-20, 20), density=True)
    if reweight_sqrt:
        target_hist = np.sqrt(target_hist)
    lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
    smoothed_value = convolve1d(target_hist, weights=lds_kernel_window, mode='constant')
    weights = [1.0/(x+1.0) for x in smoothed_value]
    np.save('weights.npy', weights)
    print('target range:', targets.min(), targets.max())
    return weights


if __name__ == '__main__':
    file_test = glob('../dataset/sige_eval_222/*.pt')
    print(file_test)
    np.save('files_test.npy', file_test)
    test_data = DatasetLDS(file_name='files_test.npy', weights='../weights.npy')
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
    for i, data in enumerate(test_loader):
        print('x' in data.keys)
