import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mnist_data import MNIST_Frames
from prednet import PredNet

from debug import info

num_epochs = 150
batch_size = 16
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
lr = 0.001  # if epoch < 75 else 0.0001
nt = 3  # num of time steps

layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cpu())
time_loss_weights = 1. / (nt - 1) * torch.ones(nt, 1)
time_loss_weights[0] = 0
time_loss_weights = Variable(time_loss_weights.cpu())

mnist_train = MNIST_Frames(nt, train=True)
mnist_val = MNIST_Frames(nt, train=False)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=True)

model = PredNet(R_channels, A_channels, output_mode='error')
if torch.cuda.is_available():
    print('Using GPU.')
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def lr_scheduler(optimizer, epoch):
    if epoch < num_epochs // 2:
        return optimizer
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        return optimizer


for epoch in range(num_epochs):
    optimizer = lr_scheduler(optimizer, epoch)
    for i, inputs, labels in enumerate(train_loader):  # TODO: Get labels from MNIST and iterate over them as well
        inputs = inputs.permute(0, 1, 4, 2, 3)  # batch x time_steps x channel x width x height
        inputs = Variable(inputs.cpu())
        errors = model(inputs)  # batch x n_layers x nt
        # errors, classification = model(inputs)  # batch x n_layers x nt
        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, nt), time_loss_weights)  # batch*n_layers x 1
        errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
        errors = torch.mean(errors)

        # TODO: Get class from MNIST and compare to classification, and subtract the classification of network from true value to get classification error
        # errors_classification = torch.nn.functional.mse_loss(classification, labels)  # Maybe change loss function to cross_entropy
        # errors_total = errors + errors_classification

        optimizer.zero_grad()

        errors.backward()  # Change to errors_total

        optimizer.step()
        if i % 2 == 0:
            print('Epoch: {}/{}, step: {}/{}, errors: {}'.format(epoch, num_epochs, i, len(kitti_train) // batch_size, errors.item()))

torch.save(model.state_dict(), 'training.pt')
