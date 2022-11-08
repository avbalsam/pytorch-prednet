import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mnist_data import MNIST_Frames
from prednet import PredNet

from debug import info

num_epochs = 10
batch_size = 16
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
lr = 0.0001  # if epoch < 75 else 0.0001
nt = 3  # num of time steps

if torch.cuda.is_available():
    print('Using GPU.')
    cuda_available = True
else:
    cuda_available = False


time_loss_weights = 1. / (nt - 1) * torch.ones(nt, 1)
time_loss_weights[0] = 0

if cuda_available:
    layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cuda())
    time_loss_weights = Variable(time_loss_weights.cuda())
else:
    layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cpu())
    time_loss_weights = Variable(time_loss_weights.cpu())


mnist_train = MNIST_Frames(nt, train=True, noise_type='gaussian', noise_intensity=0.1)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

model = PredNet(R_channels, A_channels, use_cuda=cuda_available)

if cuda_available:
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
        if cuda_available:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs.cpu())

        rec_error, classification = model(inputs)  # batch x n_layers x nt

        loc_batch = rec_error.size(0)
        rec_error = torch.mm(rec_error.view(-1, nt), time_loss_weights)  # batch*n_layers x 1
        rec_error = torch.mm(rec_error.view(loc_batch, -1), layer_loss_weights)
        rec_error = torch.mean(rec_error)

        # Create a tensor of label arrays to compare with classification tensor
        label_arr = [[float(label == labels[i]) for label in range(10)] for i in range(16)]
        class_error = list()
        for c in range(len(classification)):
            if cuda_available:
                label_tensor = torch.cuda.FloatTensor(label_arr[c])
            else:
                label_tensor = torch.FloatTensor(label_arr[c])
            classification_loss = torch.nn.functional.cross_entropy(classification[c], label_tensor)
            class_error.append(classification_loss)

        mean_class_error = sum(class_error) / len(class_error)
        errors_total = (0.8 * rec_error) + (0.2 * mean_class_error)

        optimizer.zero_grad()

        errors_total.backward()

        optimizer.step()
        if i % 2 == 0:
            print('Epoch: {}/{}, step: {}/{}, reconstruction error: {}, classification error: {}, total error: {}'
                  .format(epoch, num_epochs, i, len(mnist_train) // batch_size, round(rec_error.item(), 3),
                          round(mean_class_error.item(), 3), round(errors_total.item(), 3)),
                  flush=True)

torch.save(model.state_dict(), './training.pt')
