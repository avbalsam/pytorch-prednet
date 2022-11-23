import argparse
import csv

import pandas as pd
import seaborn as sns

import csv

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from mnist_data_prednet import MNIST_Frames
from prednet import PredNet
from controls.prednet_additive import PredNetAdditive
from controls.prednet_feedforward import PredNetFF
from train import get_accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--dir_name', default='', type=str, help='path to directory of model to plot')
    args = parser.parse_args()

    return args


def plot_epochs(plot_type):
    valid_plot_types = ['loss', 'accuracy']
    assert plot_type in valid_plot_types, f"Please choose a valid plot type from {valid_plot_types}"

    # dirname = f"/Users/avbalsam/Documents/openmind/prednet/model_5_gaussian_1.0/"

    if plot_type in ['loss', 'accuracy']:
        filename = f"{plot_type}_log.csv"

        data = list()

        with open(f"{dir_name}/{filename}", newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                data.append(row)

        fields = data.pop(0)

        data = [[float(i) for i in row] for row in data]

        sns.set_theme()

        data = pd.DataFrame(data[1:], columns=fields)

        dfm = data.melt('Epochs', var_name=f'{plot_type} type', value_name=plot_type)

        return sns.relplot(
            data=dfm, kind="line",
            x="Epochs", y=plot_type, hue=f'{plot_type} type'
        )


def plot_timesteps(plot_type, nt):
    valid_plot_types = ['timestep accuracy']
    assert plot_type in valid_plot_types, f"Please pick a valid plot type from {valid_plot_types}"
    data = [["Timestep", "Accuracy"]]
    for timestep in range(nt):
        accuracy = get_accuracy(test_loader, model, timestep)
        print(f"Timestep: {timestep}, Accuracy: {accuracy}")
        data.append([timestep, accuracy])

    data = pd.DataFrame(data[1:], columns=data[0])

    return sns.relplot(
        data=data, kind="line",
        x="Timestep", y="Accuracy"
    )


def plot_noise_levels(model, nt, noise_type='gaussian', noise_levels=None):
    accuracy_over_noise = [["Noise level", "Accuracy", "Timestep"]]
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3]
    for timestep in range(nt):
        for level in noise_levels:
            mnist_noise = MNIST_Frames(nt, train=False, noise_type=noise_type, noise_intensity=level)
            val_loader = DataLoader(mnist_noise, batch_size=batch_size, shuffle=True)
            accuracy_over_noise.append([level, get_accuracy(val_loader, model, timestep), timestep])

    data = pd.DataFrame(accuracy_over_noise[1:], columns=accuracy_over_noise[0])
    return sns.relplot(
        data=data, kind="line",
        x="Noise level", y="Accuracy", hue="Timestep"
    )


if __name__ == "__main__":
    args = parse_args()

    plot_types = ['loss', 'accuracy', 'timestep accuracy']

    model_name = 'model.pt'
    models = [PredNet, PredNetAdditive]

    batch_size = 16
    A_channels = (3, 48, 96, 192)
    R_channels = (3, 48, 96, 192)

    nt = 5

    for model in models:
        mnist_test = MNIST_Frames(nt, train=False)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
        model_to_plot = model(R_channels, A_channels, nt).to(device)
        
        dir_name = model_to_plot.get_name() if args.dir_name == '' else args.dir_name

        print(f"Plotting data for model {dir_name}...")
        model_to_plot.load_state_dict(torch.load(f"{dir_name}/{model_name}", map_location=device))

        print(f"Plotting loss and accuracy over epochs for model {dir_name}...")
        plot_epochs('loss').savefig(f"{dir_name}/loss_plot.png")
        plot_epochs('accuracy').savefig(f"{dir_name}/accuracy_plot.png")

        print(f"Plotting accuracy over noise levels for model {dir_name}...")
        plot_noise_levels(model_to_plot, nt).savefig(f"{dir_name}/noise_level_accuracy_plot.png")

        print(f"Plotting accuracy over timestep for model {dir_name}...")
        plot_timesteps('timestep accuracy', nt).savefig(f"{dir_name}/timestep_accuracy_plot.png")

        print(f"Finished plotting {dir_name}!\n\n")
