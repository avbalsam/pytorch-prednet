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
        img = torchvision.transforms.Resize(32)(item[0])
        frames = img.unsqueeze(0)
        frames = frames.repeat(self.nt, 3, 1, 1)
        _class = item[1]
        return frames, _class

    def __len__(self):
        return len(super().train_data)

