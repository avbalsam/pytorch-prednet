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
lr = 0.00001  # if epoch < 75 else 0.0001
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
    for i, (inputs, labels) in enumerate(train_loader):
        # inputs = inputs.permute(0, 1, 4, 2, 3)  # batch x time_steps x channel x width x height
        inputs = Variable(inputs.cpu())

        # errors = model(inputs)  # batch x n_layers x nt
        errors, classification = model(inputs)  # batch x n_layers x nt

        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, nt), time_loss_weights)  # batch*n_layers x 1
        errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
        errors = torch.mean(errors)

        label_arr = torch.FloatTensor([[float(label == labels[i]) for label in range(16)] for i in range(16)]) # Create a tensor of label arrays to compare with classification tensor
        classification_error = torch.nn.functional.mse_loss(classification, label_arr)  # Maybe change loss function to cross_entropy

        reconstruction_error = errors
        errors_total = torch.add(errors, classification_error.item())

        optimizer.zero_grad()

        errors.backward()  # Change to errors_total

        optimizer.step()
        if i % 2 == 0:
            print('Epoch: {}/{}, step: {}/{}, reconstruction error: {}, classification error: {}, total error: {}'.format(epoch, num_epochs, i, len(mnist_train) // batch_size, round(reconstruction_error.item(), 3), round(classification_error.item(), 3), round(errors_total.item(), 3)))

torch.save(model.state_dict(), 'training.pt')
