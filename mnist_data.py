import hickle as hkl
import numpy as np

import torch
import torch.utils.data as data
import torchvision.datasets
from torchvision.transforms import ToTensor


class MNIST_Frames(torchvision.datasets.MNIST):
    def __init__(self, nt, train: bool, root: str = "./", download: bool = True):
        super().__init__(root, train=train, download=download, transform=ToTensor())
        self.nt = nt

    def __getitem__(self, index):
        item = super().__getitem__(index)
        print(item)
        frames, _class = [np.array(item[0].getdata()) for _ in range(self.nt)], item[1]
        return frames

    def __len__(self):
        return len(super().train_data)

