import os
import random

import hickle as hkl

import torch
import torch.utils.data as data
import torchvision.datasets
from PIL.Image import Image

SHOW_PLOT = False


def get_ck_data(source_dir="/Users/avbalsam/Desktop/Predictive_Coding_UROP/CK+/cohn-kanade-ko_data",
                label_dir="/Users/avbalsam/Desktop/Predictive_Coding_UROP/CK+/Emotion",
                output_path="/om2/user/avbalsam/prednet/ck_data/ck_data.hkl"):
    ck_data = list()
    for subject in os.listdir(source_dir):
        subject_path = f"{source_dir}/{subject}"
        if not os.path.isfile(subject_path):
            for num in os.listdir(subject_path):
                source_path = f"{subject_path}/{num}"
                if not os.path.isfile(source_path):
                    frames = list()
                    for frame_file in os.listdir(source_path):
                        if frame_file.endswith(".png"):
                            # Resize image and convert to RGB
                            img = torchvision.io.read_image(f"{source_path}/{frame_file}")
                            if img.size(dim=0) == 1:
                                rgb_like = torchvision.transforms.functional.rgb_to_grayscale(img.repeat(3, 1, 1))
                            else:
                                rgb_like = img
                            resized = torchvision.transforms.Resize((512, 512))(rgb_like)
                            img = torch.unsqueeze(resized, 0)
                            frames.append(img[0].numpy())
                            # print(f"Processed file {source_path}/{frame_file}...")

                    if os.path.exists(f"{label_dir}/{subject}/{num}"):
                        for filename in os.listdir(f"{label_dir}/{subject}/{num}"):
                            if filename.endswith(".txt"):
                                with open(f"{label_dir}/{subject}/{num}/{filename}", "r") as label_file:
                                    label = int(label_file.read().strip().split(".")[0])
                                    print(f"Finished frame sequence {source_path}. Label was {label}.")
                                    ck_data.append((frames, label))
                                    break
                        else:
                            print(
                                f"Could not find label for {source_path}. It will not be added to training data.")

    hkl.dump(ck_data, output_path, mode='w', compression='gzip')
    return ck_data


class FrameSequenceDataset(data.Dataset):
    """General superclass for any dataset which includes only frame sequences"""
    def __init__(self, raw_data, nt: int, train: bool, transforms=None):
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7]

        self.half = None
        self.transforms = transforms

        classes = dict()
        for frames, label in raw_data:
            if label in classes:
                classes[label].append((frames, label))
            else:
                classes[label] = [(frames, label)]

        for _class in classes.values():
            random.shuffle(_class)
        class_count = {key: int(len(value) * (0.80 if train else 0.20)) for (key, value) in classes.items()}

        self.data = list()
        for key in classes.keys():
            while class_count[key] > 0:
                self.data.append(classes[key].pop(0))
                class_count[key] -= 1

        self.nt = nt

    def get_half(self):
        return self.half

    def set_half(self, half: str):
        """
        Choose a half of the image to use. The rest will be blacked out. If self.half is set to None (which it is by
        default), the whole image will be used.

        :param half: Either 'top' or 'bottom'
        :return: None
        """
        self.half = half

    def get_labels(self):
        """
        Returns a list of all possible labels for this dataset.
        """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        frames, label = self.data[index]

        # Repeat the first frame if sequence is too short
        while len(frames) < self.nt:
            frames.insert(0, frames[0])
        frames = frames[self.nt * -1:]

        # Replace all frames with last frame in sequence
        for i in range(len(frames)):
            frames[i] = frames[-1]

        frames_transformed = list()
        state = None
        for frame in frames:
            image = torchvision.transforms.Resize((256, 256))(torch.from_numpy(frame).unsqueeze(0))

            # If we are supposed to get the top or bottom half of the image, do that now
            if self.half is not None:
                if self.half == 'top':
                    image = torch.cat(
                        (
                            image[:, :, :int(image.shape[2] / 2), :],
                            torch.zeros(image.shape[0], image.shape[1], int(image.shape[2] / 2), image.shape[3])
                        ), dim=2)
                elif self.half == 'bottom':
                    image = torch.cat(
                        (
                            torch.zeros(image.shape[0], image.shape[1], int(image.shape[2] / 2), image.shape[3]),
                            image[:, :, int(image.shape[2] / 2):, :]
                        ), dim=2)

            # Ensure that the same transformation is applied to all frames by resetting the rng state
            if state is None:
                state = torch.get_rng_state()
            else:
                torch.set_rng_state(state)
            if self.transforms is not None:
                image = self.transforms(image)

            # Append the transformed image to the list
            frames_transformed.append(image)

        # Make sure all ko_data have three channels
        if frames_transformed[0].size(dim=1) == 1:
            frames_transformed = [frame.repeat(1, 3, 1, 1) for frame in frames_transformed]
        elif frames_transformed[0].size(dim=1) == 3:
            pass
        else:
            print(f"Wrong number of channels on input tensor: {frames_transformed[0].size()}")
        frames_transformed = torch.cat(frames_transformed, 0)

        if SHOW_PLOT:
            from matplotlib import pyplot as plt
            fig, axes = plt.subplots(1, 10)
            for i, frame in enumerate(frames):
                ax = axes[i]
                ax.imshow(torch.from_numpy(frame).permute(1, 2, 0).cpu().squeeze())
                ax.axis('off')
                ax.set_title(f"{index} {i}")
            plt.show()

            fig, axes = plt.subplots(1, 10)
            for i, frame in enumerate(frames_transformed):
                ax = axes[i]
                ax.imshow(frame[0])
                ax.axis('off')
                ax.set_title(f"{index} {i}")
            plt.show()

        return frames_transformed, label


class CK(FrameSequenceDataset):
    def __init__(self, nt: int, train: bool, data_path: str = "/om2/user/avbalsam/prednet/ck_data/ck_data.hkl",
                 transforms=None):
        if os.path.exists("/Users/avbalsam/Desktop/Predictive_Coding_UROP/ck_data/ck_data.hkl"):
            raw_data = hkl.load("/Users/avbalsam/Desktop/Predictive_Coding_UROP/ck_data/ck_data.hkl")
        elif os.path.exists(data_path):
            raw_data = hkl.load(data_path)
        else:
            print("Could not find pickled ck_data file. Compiling data from scratch...")
            raw_data = get_ck_data(output_path=data_path)
        super().__init__(raw_data, nt=nt, train=train, transforms=transforms)

    def get_name(self):
        return f"CK+_{'no' if self.half is None else self.half}_half"


class CKStatic(CK):
    """Access repeated static ko_data from CK+ dataset,
    to use as a control.
    To choose which frame of the video sequences to access,
    change the n_frame parameter.
    To ensure compatibility with the prednet, the getitem
    method of this class repeats one frame self.nt times."""
    n_frame = 0

    def set_n_frame(self, n_frame):
        self.n_frame = n_frame

    def __getitem__(self, index):
        frames, label = super().__getitem__(index)
        return frames[self.n_frame].repeat(self.nt, 1, 1, 1), label

    def get_name(self):
        return f"{super().get_name()}_frame_{self.n_frame}"


class Psychometric(FrameSequenceDataset):
    def __init__(self, nt: int, train: bool, transforms=None, data_path='./ko_data'):
        data = []
        for filename in os.listdir(data_path):
            if filename[0] == '.':
                continue
            words = filename.split(".")[0].split('-')
            fear = int(words[1].replace("fe", ""))
            happiness = int(words[2].replace("ha", ""))
            img = torchvision.io.read_image(f"{data_path}/{filename}")
            if img.size(dim=0) == 1:
                rgb_like = torchvision.transforms.functional.rgb_to_grayscale(img.repeat(3, 1, 1))
            else:
                rgb_like = img
            resized = torchvision.transforms.Resize((256, 256))(rgb_like)
            # img = torch.unsqueeze(resized, 0).repeat(1, 3, 1, 1).numpy()
            img = resized.numpy()
            frames = list()
            for _ in range(nt):
                frames.append(img)

            # frames = torch.cat(frames, 0)

            # 0: Fear, 1: Happiness
            data.append((frames, 0 if fear > happiness else 1))

        super().__init__(data, nt=nt, train=train, transforms=None)

    def get_name(self):
        return f"Psychometric_{'no' if self.half is None else self.half}_half"


if __name__ == "__main__":
    pass
    # /om2/user/avbalsam/prednet/ck_data/ck_data.hkl
    # /Users/avbalsam/Desktop/Predictive_Coding_UROP/prednet/ck_data/ck_data.hkl
    # Image.show(torchvision.transforms.ToPILImage()(torch.from_numpy(c[0][0])))
