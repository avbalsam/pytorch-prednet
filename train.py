import argparse
import csv
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from prednet import PredNet
from mnist_data_prednet import MNIST_Frames


MODELS = {'prednet': PredNet}
DATASETS = {'mnist_frames': MNIST_Frames}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', default=1, type=int, help='experiment version')
    # parser.add_argument('-j', '--job', default=1, type=int, help='slurm array job id')
    # parser.add_argument('-i', '--id', default=1, type=int, help='slurm array task id')
    parser.add_argument('-s', '--is_slurm', default=True, type=bool, help='use slurm')
    parser.add_argument('-m', '--model_name', default='prednet', type=str, help='filename of python file for model')
    parser.add_argument('-d', '--data_name', default='mnist_frames', type=str,
                        help='filename of python file for dataset')
    parser.add_argument('-n', '--noise', default=0.0, type=float,
                        help='amount of gaussian noise to add to dataset images')
    parser.add_argument('-b', '--blur', default=0.0, type=float, help='amount of blur to add to dataset images')
    # parser.add_argument('-c', '--csv_path', default='./', type=str, help='csv path')
    args = parser.parse_args()

    return args


def get_accuracy(val_loader, model, device):
    correct_guesses = 0
    total_guesses = 0
    accuracy = 0
    wrong_guesses = list()
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = Variable(inputs.to(device))

        rec_error, classification = model(inputs)

        for j in range(len(classification)):
            c = classification[j].tolist()
            _class = c.index(max(c))
            print(
                f"Image {i + j}; Correct label: {labels[j]}; Predicted: {_class}; Correct guesses: {correct_guesses}; "
                f"Accuracy: {round(accuracy * 100, 3)}%")
            total_guesses += 1
            if labels[j] == _class:
                correct_guesses += 1
                accuracy = correct_guesses / total_guesses
            else:
                wrong_guesses.append((labels[j], _class))
    return accuracy


def torch_main(args):
    assert args.model_name in MODELS.keys(), f'Please choose a valid model name from {MODELS.keys()}'
    assert args.data_name in DATASETS.keys(), f'Please choose a valid dataset from {DATASETS.keys()}'

    if torch.cuda.is_available():
        print('Using GPU.')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.model_name == 'prednet':

        # Weight to give to reconstruction and classification error when calculating total error
        rec_weight = 0.9
        class_weight = 0.1

        num_epochs = 50
        batch_size = 16
        A_channels = (3, 48, 96, 192)
        R_channels = (3, 48, 96, 192)
        lr = 0.0001  # if epoch < 75 else 0.0001
        nt = 5  # num of time steps

        noise_intensity = args.noise
        noise_type = 'gaussian'

        print(f"Epochs: {num_epochs}\n"
              f"Learning rate: {lr}\n"
              f"Time steps: {nt}\n"
              f"Reconstruction weight: {rec_weight}\n"
              f"Classification weight: {class_weight}\n"
              f"Noise type: {noise_type}\n"
              f"Noise intensity: {noise_intensity}\n\n\n", flush=True)

        dir_name = f"model_{nt}_{class_weight}_{rec_weight}_{noise_type}_{noise_intensity}"

        if os.path.exists(f"./{dir_name}"):
            dir_name += "_(1)"
            os.mkdir(f"./{dir_name}")
        else:
            os.mkdir(f"./{dir_name}")

        def lr_scheduler(optimizer, epoch):
            if epoch < num_epochs // 2:
                return optimizer
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0001
                return optimizer

        train_dataset = DATASETS[args.data_name](nt, train=True, noise_type=noise_type, noise_intensity=noise_intensity)
        val_dataset = DATASETS[args.data_name](nt, train=False, noise_type=noise_type, noise_intensity=noise_intensity)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        model = PredNet(R_channels=R_channels, A_channels=A_channels, device=device, nt=nt,
                        class_weight=class_weight, rec_weight=rec_weight)

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_acc_epochs = list()
    val_acc_epochs = list()
    loss_epochs = list()

    for epoch in range(num_epochs):
        accuracy = 0
        total_guesses = 0
        correct_guesses = 0
        optimizer = lr_scheduler(optimizer, epoch)

        for i, (inputs, labels) in enumerate(train_loader):
            # inputs = inputs.permute(0, 1, 4, 2, 3)  # batch x time_steps x channel x width x height
            inputs = Variable(inputs.to(device))

            model_output = model(inputs)  # batch x n_layers x nt

            loss = model.calculate_loss(model_output, labels)

            rec_error, classification = model_output
            for j in range(len(classification)):
                c = classification[j].tolist()
                _class = c.index(max(c))
                total_guesses += 1
                if _class == labels[j]:
                    correct_guesses += 1
                print(f"Total guesses: {total_guesses}, Correct guesses: {correct_guesses}")

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if i % 10 == 0:
                print('Epoch: {}/{}, Step: {}/{}, Loss: {}'
                      .format(epoch, num_epochs, i, len(train_dataset) // batch_size, round(loss.item(), 7)),
                      flush=True)

        loss_epochs.append(loss.item())
        train_accuracy = correct_guesses / total_guesses
        training_acc_epochs.append(train_accuracy)
        val_accuracy = get_accuracy(val_loader, model, device)
        val_acc_epochs.append(val_accuracy)

        with open(f'./{dir_name}/log.csv', 'w') as f:

            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(["Training accuracy", "Validation accuracy", "Total loss"])
            rows = [training_acc_epochs, val_acc_epochs, loss_epochs]
            write.writerows(rows)

        print('\n\n\n\nEpoch: {}/{}, Train accuracy: {}%, Validation accuracy: {}%'
              .format(epoch, num_epochs, round(train_accuracy * 100, 3), round(val_accuracy * 100, 3)),
              flush=True)

    torch.save(model.state_dict(),
               f'./{dir_name}/model.pt')


if __name__ == '__main__':
    args = parse_args()
    torch_main(args)
