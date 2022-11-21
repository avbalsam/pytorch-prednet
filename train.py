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


def generate_model_name(nt, noise_type='gaussian', noise_intensity=0.0, version=1):
    return f"model_{nt}_{noise_type}_{noise_intensity}_v{version}"


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
    args = parser.parse_args()

    return args


def get_accuracy(val_loader, model, device, timestep=None):
    correct_guesses = 0
    total_guesses = 0
    accuracy = 0
    wrong_guesses = list()
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = Variable(inputs.to(device))

        if timestep is None:
            rec_error, classification = model(inputs)
        else:
            rec_error, classification = model(inputs, timestep)

        for j in range(len(classification)):
            c = classification[j].tolist()
            _class = c.index(max(c))
            # print(
            #    f"Image {i + j}; Correct label: {labels[j]}; Predicted: {_class}; Correct guesses: {correct_guesses}; "
            #    f"Accuracy: {round(accuracy * 100, 3)}%")
            total_guesses += 1
            if labels[j] == _class:
                correct_guesses += 1
                accuracy = correct_guesses / total_guesses
            else:
                wrong_guesses.append((labels[j], _class))

        if i % 10 == 0:
            print("Timestep: {}, Step: {}, Accuracy: {}".format("avg" if timestep is None else timestep, i, accuracy))
    return accuracy


def torch_main(args):
    assert args.model_name in MODELS.keys(), f'Please choose a valid model name from {MODELS.keys()}'
    assert args.data_name in DATASETS.keys(), f'Please choose a valid dataset from {DATASETS.keys()}'

    version = args.version

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

        # For exclusively feedforward network, use nt=1
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

        dir_name = generate_model_name(nt=nt, noise_type=noise_type, noise_intensity=noise_intensity, version=version)

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
    class_error_epochs = list()
    rec_error_epochs = list()

    for epoch in range(num_epochs):
        accuracy = 0
        total_guesses = 0
        correct_guesses = 0
        optimizer = lr_scheduler(optimizer, epoch)

        # TODO: Find some way to make this completely model independent
        for i, (inputs, labels) in enumerate(train_loader):
            # inputs = inputs.permute(0, 1, 4, 2, 3)  # batch x time_steps x channel x width x height
            inputs = Variable(inputs.to(device))

            model_output = model(inputs)  # batch x n_layers x nt

            loss = model.calculate_loss(model_output, labels)

            rec_error, classification = model_output
            classification_error = 10*(loss - 0.9 * rec_error)
            for j in range(len(classification)):
                c = classification[j].tolist()
                _class = c.index(max(c))
                total_guesses += 1
                if _class == labels[j]:
                    correct_guesses += 1

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if i % 10 == 0:
                print('Epoch: {}/{}, Step: {}/{}, Rec Loss: {}, Class Loss: {}'
                      .format(epoch, num_epochs, i, len(train_dataset) // batch_size, round(rec_error.item(), 7),
                      round(classification_error.item(), 7)),
                      flush=True)

        # Add loss and accuracy from this epoch to list
        class_error_epochs.append(classification_error.item())
        rec_error_epochs.append(rec_error.item())
        train_accuracy = correct_guesses / total_guesses
        training_acc_epochs.append(train_accuracy)
        val_accuracy = get_accuracy(val_loader, model, device)
        val_acc_epochs.append(val_accuracy)

        if not os.path.exists(f"./{dir_name}"):
            os.mkdir(f"./{dir_name}")
        # Write accuracy data to csv file
        with open(f'./{dir_name}/accuracy_log.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(["Epochs", "Training accuracy", "Validation accuracy"])
            for i in range(len(training_acc_epochs)):
                write.writerow([i, float(training_acc_epochs[i]), float(val_acc_epochs[i])])

        # Write loss data to csv file
        with open(f'./{dir_name}/loss_log.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(["Epochs", "Classification Error", "Reconstruction Error"])
            for i in range(len(training_acc_epochs)):
                write.writerow([i, float(class_error_epochs[i]), float(rec_error_epochs[i])])

        print('\n\n\n\nEpoch: {}/{}, Train accuracy: {}%, Validation accuracy: {}%'
              .format(epoch, num_epochs, round(train_accuracy * 100, 3), round(val_accuracy * 100, 3)),
              flush=True)

    torch.save(model.state_dict(),
               f'./{dir_name}/model.pt')


if __name__ == '__main__':
    args = parse_args()
    torch_main(args)
