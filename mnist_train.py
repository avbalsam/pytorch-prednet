import os

import hickle
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mnist_data import MNIST_Frames
from prednet import PredNet

from debug import info

import sys
import getopt

noise_type = "gaussian"
noise_intensity = 0.0

# Weight to give to reconstruction and classification error when calculating total error
rec_weight = 0.9
class_weight = 0.1

num_epochs = 50
batch_size = 16
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
lr = 0.0001  # if epoch < 75 else 0.0001
nt = 3  # num of time steps

# Parse command line arguments to get specifications for model
arg_help = "{0} -e <num_epochs> -t <num_timesteps> -l <learning_rate> -c <classification_weight> -r <reconstruction_weight> -n <noise_amt>".format(
    sys.argv[0])

try:
    opts, args = getopt.getopt(sys.argv[1:], "he:t:l:c:r:n:", ["help", "epochs=",
                                                               "timesteps=", "learning_rate=", "class_weight=",
                                                               "rec_weight=", "noise_amt="])
except:
    print(arg_help)
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print(arg_help)  # print the help message
        sys.exit(2)
    elif opt in ("-e", "--epochs"):
        num_epochs = int(arg)
    elif opt in ("-t", "--timesteps"):
        nt = int(arg)
    elif opt in ("-l", "--learning_rate"):
        lr = float(arg)
    elif opt in ("-c", "--class_weight"):
        class_weight = float(arg)
    elif opt in ("-r", "--rec_weight"):
        rec_weight = float(arg)
    elif opt in ("-n", "--noise_amt"):
        noise_intensity = float(arg)

if torch.cuda.is_available():
    print('Using GPU.')
    cuda_available = True
else:
    cuda_available = False

print(f"Epochs: {num_epochs}\n"
      f"Learning rate: {lr}\n"
      f"Time steps: {nt}\n"
      f"Reconstruction weight: {rec_weight}\n"
      f"Classification weight: {class_weight}\n\n\n", flush=True)

time_loss_weights = 1. / (nt - 1) * torch.ones(nt, 1)
time_loss_weights[0] = 0

if cuda_available:
    layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cuda())
    time_loss_weights = Variable(time_loss_weights.cuda())
else:
    layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cpu())
    time_loss_weights = Variable(time_loss_weights.cpu())

mnist_train = MNIST_Frames(nt, train=True, noise_type='gaussian', noise_intensity=0.0)

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


accuracy_over_epochs = list()
for epoch in range(num_epochs):
    accuracy = 0
    total_guesses = 0
    correct_guesses = 0
    optimizer = lr_scheduler(optimizer, epoch)

    for i, (inputs, labels) in enumerate(train_loader):
        # inputs = inputs.permute(0, 1, 4, 2, 3)  # batch x time_steps x channel x width x height
        if cuda_available:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs.cpu())

        rec_error, classification = model(inputs)  # batch x n_layers x nt

        # Update classification accuracy
        for j in range(len(classification)):
            c = classification[j].tolist()
            _class = c.index(max(c))
            total_guesses += 1
            if _class == labels[j]:
                correct_guesses += 1

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
        errors_total = (rec_weight * rec_error) + (class_weight * mean_class_error)

        optimizer.zero_grad()

        errors_total.backward()

        optimizer.step()
        if i % 10 == 0:
            print('Epoch: {}/{}, step: {}/{}, reconstruction error: {}, classification error: {}, total error: {}'
                  .format(epoch, num_epochs, i, len(mnist_train) // batch_size, round(rec_error.item(), 7),
                          round(mean_class_error.item(), 7), round(errors_total.item(), 7)),
                  flush=True)

    accuracy = correct_guesses / total_guesses
    accuracy_over_epochs.append(accuracy)
    print('\n\n\n\nEpoch: {}/{}, Accuracy: {}%'
          .format(epoch, num_epochs, round(accuracy * 100, 2)),
          flush=True)

dir_name = f"model_{num_epochs}_{batch_size}_{lr}_{nt}_{class_weight}_{rec_weight}_{noise_type}_{noise_intensity}"

if os.path.exists(f"./{dir_name}"):
    print("Redundant model has not been saved.")
else:
    os.mkdir(f"./{dir_name}")

torch.save(model.state_dict(),
           f'./{dir_name}/model_{num_epochs}_{batch_size}_{lr}_{nt}_{class_weight}_{rec_weight}_{noise_type}_{noise_intensity}.pt')

plt.plot(accuracy_over_epochs)
plt.savefig(f"./{dir_name}/accuracy_plot.png")
