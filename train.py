import argparse
import csv
import os

import torch
import torchvision.transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import models
from plot_data import plot
from utility import get_accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='prednet', type=str, help='filename of python file for model')
    parser.add_argument('-d', '--data_name', default='psych', type=str,
                        help='filename of python file for dataset')
    args = parser.parse_args()

    return args


def torch_main(args):
    if torch.cuda.is_available():
        print('Using GPU.')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if 'prednet' in args.model_name:
        num_epochs = 150
        batch_size = 16
        lr = 0.0001  # if epoch < 75 else 0.0001

        # For exclusively feedforward network, use nt=1

        model = models.get_model_by_name(args.model_name)

        nt = model.nt

        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomAffine(degrees=15, translate=[0, 0.3], scale=[0.7, 1.15]),
            ]
        )

        train_dataset = models.get_dataset_by_name(name=args.data_name, nt=nt,
                                                   train=True, transforms=transforms, half=None)
        val_dataset = models.get_dataset_by_name(name=args.data_name, nt=nt,
                                                 train=False, transforms=None, half=None)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        model.to(device)
        dir_name = f"model:{model.get_name()}:dataset:{train_dataset.get_name()}"

        print(f"Model: {dir_name}\n"
              f"Epochs: {num_epochs}\n"
              f"Learning rate: {lr}\n"
              f"Time steps: {nt}\n\n\n", flush=True)

        def lr_scheduler(optimizer, epoch):
            if epoch < num_epochs // 2:
                return optimizer
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0001
                return optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_acc_epochs = list()
    val_acc_epochs = list()
    class_error_epochs = list()
    rec_error_epochs = list()

    for epoch in range(num_epochs):
        total_guesses = 0
        correct_guesses = 0
        optimizer = lr_scheduler(optimizer, epoch)

        for i, (inputs, labels) in enumerate(train_loader):
            # inputs = inputs.permute(0, 1, 4, 2, 3)  # batch x time_steps x channel x width x height
            inputs = Variable(inputs.to(device))

            classification = model(inputs)  # batch x n_layers x nt

            model.calculate_loss(classification, labels)

            loss = model.get_total_error()

            for j in range(len(classification)):
                c = classification[j].tolist()
                _class = c.index(max(c))
                total_guesses += 1
                if _class == labels[j]:
                    correct_guesses += 1

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            class_error = model.get_class_error()
            rec_error = model.get_rec_error()

            if i % 10 == 0:
                print('Epoch: {}/{}, Step: {}/{}, Rec Loss: {}, Class Loss: {}'
                      .format(epoch, num_epochs, i, len(train_dataset) // batch_size, round(rec_error.item(), 7),
                              round(class_error.item(), 7)),
                      flush=True)

        # Add loss and accuracy from this epoch to list
        class_error_epochs.append(class_error.item())
        rec_error_epochs.append(rec_error.item())

        train_accuracy = correct_guesses / total_guesses
        training_acc_epochs.append(train_accuracy)
        val_accuracy = get_accuracy(val_loader, model)
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
    print(f"Finished training model {dir_name}. Plotting data...")

    # plot(dir_name, model, val_dataset)


if __name__ == '__main__':
    args = parse_args()
    torch_main(args)
