import os
import hickle as hkl

import torch
import torch.utils.data as data
import torchvision.datasets
from PIL.Image import Image


def get_ck_data(source_dir="/Users/avbalsam/Desktop/Predictive_Coding_UROP/CK+/cohn-kanade-images",
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
                                rgb_like = torchvision.transforms.functional.rgb_to_grayscale(img.repeat(3,1,1))
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


class CK:
    def __init__(self, nt: int, train: bool, data_path: str = "/om2/user/avbalsam/prednet/ck_data/ck_data.hkl", noise_type=None,
                 noise_intensities=None):
        assert noise_type is None or noise_intensities is None or noise_intensities == [0.0], "Adding noise is not supported on CK dataset."
        if os.path.exists(data_path):
            self.data = hkl.load(data_path)
        else:
            print("Could not find pickled ck_data file. Compiling data from scratch...")
            self.data = get_ck_data(output_path=data_path)

        if train:
            self.data = self.data[:int(len(self.data)*(7/8))]
        else:
            self.data = self.data[int(len(self.data)*(7/8)):]
        self.nt = nt

    def __getitem__(self, index):
        frames, label = self.data[index]

        # Repeat the first frame if sequence is too short
        while len(frames) < self.nt:
            frames.insert(0, frames[0])
        frames = frames[self.nt * -1:]

        # Make sure all images have three channels
        frames = [torch.from_numpy(frame).unsqueeze(0) for frame in frames]
        if frames[0].size(dim=1) == 1:
            frames = [frame.repeat(1,3,1,1) for frame in frames]
        elif frames[0].size(dim=1) == 3:
            pass
        else:
            print(f"Wrong number of channels on input tensor: {frames[0].size()}")
        frames = torch.cat(frames, 0)

        return frames, label

    def __len__(self):
        return len(self.data)


class CKStatic(CK):
    def __getitem__(self, index):
        return super().__getitem__(index)[0].repeat(self.nt, 3, 1, 1)


if __name__ == "__main__":
    # /om2/user/avbalsam/prednet/ck_data/ck_data.hkl
    # /Users/avbalsam/Desktop/Predictive_Coding_UROP/prednet/ck_data/ck_data.hkl
    c = CK(nt=10, train=True)
    Image.show(torchvision.transforms.ToPILImage()(torch.from_numpy(c[0][0])))

