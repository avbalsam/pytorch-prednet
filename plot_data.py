import argparse
import csv

import pandas as pd
import seaborn as sns

import csv

import torch
from torch.utils.data import DataLoader

from models import MODELS, DATASETS
from utility import get_accuracy


def plot_epochs(plot_type, dir_name):
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


def plot_timesteps(model, dataset, plot_type):
    nt = model.nt

    valid_plot_types = ['timestep accuracy']
    assert plot_type in valid_plot_types, f"Please pick a valid plot type from {valid_plot_types}"
    data = [["Timestep", "Accuracy"]]
    for timestep in range(nt):
        ds = dataset(nt, train=False, noise_type=model.noise_type, noise_intensities=model.noise_intensities)
        val_loader = DataLoader(ds, batch_size=16, shuffle=True)
        accuracy = get_accuracy(val_loader, model, timestep)
        print(f"Timestep: {timestep}, Accuracy: {accuracy}")
        data.append([timestep, accuracy])

    data = pd.DataFrame(data[1:], columns=data[0])

    return sns.relplot(
        data=data, kind="line",
        x="Timestep", y="Accuracy"
    )


def plot_noise_levels(model, dataset, noise_type='gaussian', noise_levels=None):
    nt = model.nt

    accuracy_over_noise = [["Noise level", "Accuracy", "Timestep"]]
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for timestep in range(nt):
        for level in noise_levels:
            noisy_data = dataset(nt, train=False, noise_type=noise_type, noise_intensities=[level])
            val_loader = DataLoader(noisy_data, batch_size=16, shuffle=True)
            accuracy_over_noise.append([level, get_accuracy(val_loader, model, timestep), timestep])

    data = pd.DataFrame(accuracy_over_noise[1:], columns=accuracy_over_noise[0])
    return sns.relplot(
        data=data, kind="line",
        x="Noise level", y="Accuracy", hue="Timestep"
    )


def plot(model, dataset):
    dir_name = model.get_name()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.load_state_dict(torch.load(f"{dir_name}/model.pt", map_location=device))

    print(f"Plotting loss and accuracy over epochs for model {dir_name}...")
    plot_epochs('loss', dir_name).savefig(f"{dir_name}/loss_plot.png")
    plot_epochs('accuracy', dir_name).savefig(f"{dir_name}/accuracy_plot.png")

    print(f"Plotting accuracy over noise levels for model {dir_name}...")
    plot_noise_levels(model, dataset).savefig(f"{dir_name}/noise_level_accuracy_plot.png")

    print(f"Plotting accuracy over timestep for model {dir_name}...")
    plot_timesteps(model, dataset, 'timestep accuracy').savefig(
        f"{dir_name}/timestep_accuracy_plot.png")

    print(f"Finished plotting {dir_name}!\n\n")


def plot_dir(model_name, ds_name):
    model = MODELS[model_name]
    dataset = DATASETS[ds_name]

    plot(model, dataset)


if __name__ == "__main__":
    model_names = [name for name in MODELS.keys()]
    dataset_name = 'mnist_frames'
    for m in model_names:
        plot_dir(m, dataset_name)
