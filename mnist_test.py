import statistics

import torch
import os
import numpy as np
import hickle as hkl
import PIL.Image as Image

from torch.utils.data import DataLoader
from torch.autograd import Variable
from kitti_data import KITTI
from mnist_data import MNIST_Frames
from prednet import PredNet


batch_size = 16
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

nt = 3

mnist_test = MNIST_Frames(nt, train=False)

test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

model = PredNet(R_channels, A_channels)

if torch.cuda.is_available():
    print('Using GPU.')
    cuda_available = True
    model.cuda()
else:
    cuda_available = False

if cuda_available:
    model.load_state_dict(torch.load('./training.pt', map_location=torch.device('cuda')))
else:
    model.load_state_dict(torch.load('./training.pt', map_location=torch.device('cpu')))


correct_guesses = 0
total_guesses = 0
accuracy = 0
wrong_guesses = list()
for i, (inputs, labels) in enumerate(test_loader):
    if cuda_available:
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs.cpu())

    rec_error, classification = model(inputs)

    for j in range(len(classification)):
        c = classification[j].tolist()
        _class = c.index(max(c))
        print(f"Image {i+j}; Correct label: {labels[j]}; Predicted: {_class}; Correct guesses: {correct_guesses}; "
              f"Accuracy: {round(accuracy*100, 3)}%")
        total_guesses += 1
        if labels[j] == _class:
            correct_guesses += 1
            accuracy = correct_guesses / total_guesses
        else:
            wrong_guesses.append((labels[j], _class))
        break


print(f"\n\n\n\nAccuracy: {round(accuracy*100, 3)}%")
print(f"Incorrect guesses:\n\n{wrong_guesses}")