import hickle as hkl
import numpy as np

import torch
import torch.utils.data as data
import torchvision.datasets
from skimage.util import random_noise
from torchvision.transforms import ToTensor


def add_noise(image, noise_type, noise_intensity):
    assert noise_type in ['gaussian', 'localvar', 'salt', 'pepper', 's&p', 'speckle'], \
        "Incorrect noise type. Use one of ['gaussian', 'localvar', 'salt', 'pepper', 's&p', 'speckle']"
    return random_noise(image, mode=noise_type, var=noise_intensity)


class MNIST_Frames(torchvision.datasets.MNIST):
    def __init__(self, nt, train: bool, root: str = "./", download: bool = True,
                 noise_type: str = None, noise_intensity: float = 0.0):
        super().__init__(root, train=train, download=download, transform=ToTensor())
        self.nt = nt

        self.noise_type = noise_type
        self.noise_intensity = noise_intensity

    def __getitem__(self, index):
        item = super().__getitem__(index)
        img = torchvision.transforms.Resize(32)(item[0])
        if self.noise_type is not None:
            img = torch.FloatTensor(add_noise(img, noise_type=self.noise_type, noise_intensity=self.noise_intensity))
        frames = img.unsqueeze(0)
        frames = frames.repeat(self.nt, 3, 1, 1)
        _class = item[1]
        return frames, _class

    def __len__(self):
        return len(super().train_data)
