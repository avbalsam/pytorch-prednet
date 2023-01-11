import os
import pickle

import torch
import torch.utils.data as data
import torchvision.datasets


def get_ck_data(source_dir="/Users/avbalsam/Desktop/Predictive_Coding_UROP/CK+/cohn-kanade-images",
                label_dir="/Users/avbalsam/Desktop/Predictive_Coding_UROP/CK+/Emotion",
                output_path="/Users/avbalsam/Desktop/Predictive_Coding_UROP/ck_hkl/ck_data"):
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
                            frames.append(f"{source_path}/{frame_file}")
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

    with open(output_path, "wb") as outfile:
        pickle.dump(ck_data, outfile)
    return ck_data


class CK:
    def __init__(self, nt: int, train: bool = False,
                 data_path: str = "/Users/avbalsam/Desktop/Predictive_Coding_UROP/ck_hkl/ck_data.hkl", noise_type=None,
                 noise_intensities=None):
        if os.path.exists("/Users/avbalsam/Desktop/Predictive_Coding_UROP/ck_hkl/ck_data"):
            with open("/Users/avbalsam/Desktop/Predictive_Coding_UROP/ck_hkl/ck_data", "rb") as infile:
                self.data = pickle.load(infile)
        else:
            self.data = get_ck_data()
        self.nt = nt

    def __getitem__(self, index):
        frames, label = self.data[index]

        # Repeat the first frame if sequence is too short
        while len(frames) < self.nt:
            frames.insert(0, frames[0])
        # TODO: Experiment around with resizing image, and make sure proportions make sense
        frames = frames[self.nt * -1:]
        frames = [torch.unsqueeze(torchvision.transforms.Resize((128, 128))(torchvision.transforms.functional.rgb_to_grayscale(torchvision.io.read_image(frame))).repeat(3,1,1), 0) for frame in frames]
        frames = torch.cat(frames, 0)
        if frames.size() != torch.Size([10, 3, 128, 128]):
            print("Weird")
        return frames, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    c = CK(5, False)
