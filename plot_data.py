import argparse
import csv
import os.path
import shutil

import pandas
import pandas as pd
import seaborn as sns

import csv

import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from models import MODELS, DATASETS, get_model_by_name
from utility import get_accuracy

from PIL.Image import Image


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


def plot_batch_across_timesteps(model, dataset):
    nt = model.nt
    ds = dataset(nt, train=False)
    val_loader = DataLoader(ds, batch_size=4, shuffle=True)

    batch_dict = dict()

    for i, (inputs, labels) in enumerate(val_loader):
        for l in range(len(labels)):
            label = labels[l].item()
            if label not in batch_dict.keys():
                batch_dict[label] = inputs[l]

    labels = ds.get_labels()

    data = [['timestep', 'predicted_label', 'confidence']]

    # if os.path.exists(f"./{model.get_name()}/input_img_over_timesteps/"):
    #     shutil.rmtree(f"./{model.get_name()}/input_img_over_timesteps/")

    for label in labels:
        if label not in batch_dict:
            continue
        input = batch_dict[label]
        batch_list = [input] * 16
        batch_tensor = torch.stack(batch_list)
        for timestep in range(nt):
            print(f"Plotting batch on timestep {timestep}...")
            classification = model(batch_tensor, timestep)
            _class = classification[0].detach().cpu().numpy()
            _class = [i - min(_class) for i in _class]
            for c in range(len(_class)):
                # ['timestep', 'predicted_label', 'confidence']
                if c == len(labels):
                    print(f"{c}: {labels} {_class}")
                else:
                    data.append([timestep, sorted(labels)[c], _class[c]])

            output_path = f"./{model.get_name()}/input_img_over_timesteps/label_{label}"
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            plot = sns.barplot(x=sorted(labels), y=_class)
            print(f"{output_path}/timestep_{timestep}.png")
            plt.imsave(f"{output_path}/input_image_{label}.png", input[0][0], cmap='gray')
            plot.set_title(f"{model.get_name()} step {timestep} label {label}")
            plt.savefig(f"{output_path}/timestep_{timestep}.png")
            plt.show()

        # The following code creates a FacetGrid with timestep data. I found that it didn't look as good.
        """
        df = pd.DataFrame(data[1:], columns=data[0])
        grid = sns.FacetGrid(df, col="timestep")
        grid.map(sns.barplot, "predicted_label", "confidence", errorbar=None)
        plt.savefig(f"{output_path}/grid_label_{label}.png")
        plt.clf()
        print(df)
        """


def plot_timesteps(model, dataset, plot_type):
    nt = model.nt

    valid_plot_types = ['timestep accuracy']
    assert plot_type in valid_plot_types, f"Please pick a valid plot type from {valid_plot_types}"
    data = [["Timestep", "Accuracy"]]
    ds = dataset(nt, train=False)
    val_loader = DataLoader(ds, batch_size=4, shuffle=True)
    for timestep in range(nt):
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

    plot_batch_across_timesteps(model, dataset)

    print(f"Plotting loss and accuracy over epochs for model {dir_name}...")
    plot_epochs('loss', dir_name).savefig(f"{dir_name}/loss_plot.png")
    plot_epochs('accuracy', dir_name).savefig(f"{dir_name}/accuracy_plot.png")

    # print(f"Plotting accuracy over noise levels for model {dir_name}...")
    # plot_noise_levels(model, dataset).savefig(f"{dir_name}/noise_level_accuracy_plot.png")

    print(f"Plotting accuracy over timestep for model {dir_name}...")
    plot_timesteps(model, dataset, 'timestep accuracy').savefig(
        f"{dir_name}/timestep_accuracy_plot.png")

    print(f"Finished plotting {dir_name}!\n\n")


def plot_dir(model_name, ds_name):
    model = MODELS[model_name]
    dataset = DATASETS[ds_name]

    plot(model, dataset)


def show_sample_input(dataset, nt):
    data = dataset(nt, train=True)
    data_loader = DataLoader(data, batch_size=16, shuffle=True)
    for (label, frames) in enumerate(data_loader):
        for i, frame in enumerate(frames[0][0]):
            img = torchvision.transforms.ToPILImage()(frame[0])
            Image.show(img)
        break


if __name__ == "__main__":
    # show_sample_input(DATASETS['CK'], nt=10)
    plot(get_model_by_name('prednet', class_weight=0.9, rec_weight=0.1, nt=10, noise_type='gaussian',
                           noise_intensities=[0.0]), DATASETS['CK'])
    print("Finished plotting prednet no noise\n\n\n", flush=True)
    plot(get_model_by_name('prednet', class_weight=0.1, rec_weight=0.9, nt=10, noise_type='gaussian',
                           noise_intensities=[0.0, 0.25, 0.5]), DATASETS['mnist_frames'])
    print("Finished plotting prednet with noise\n\n\n", flush=True)
    plot(get_model_by_name('prednet_additive', class_weight=0.1, rec_weight=0.9, nt=10, noise_type='gaussian',
                           noise_intensities=[0.0, 0.25, 0.5]), DATASETS['mnist_frames'])
    # exit(0)
    # model_names = [name for name in MODELS.keys()]
    # dataset_name = 'mnist_frames'
    # for m in model_names:
    #    plot_dir(m, dataset_name)
