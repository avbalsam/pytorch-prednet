import torch

from ck_data import CK, CKStatic
from controls.prednet_additive import PredNetAdditive
from controls.prednet_feedforward import PredNetFF
from mnist_data_prednet import MNIST_Frames
from prednet import PredNet

A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
nt = 10  # Number of timesteps
NOISE_TYPE = 'gaussian'
NOISE_INTENSITIES = [0.0]


# NOISE_INTENSITIES = [0.0]


def get_model_by_name(model_name, nt=10, class_weight=0.9, rec_weight=0.1):
    if model_name == 'prednet':
        return PredNet(R_channels=R_channels, A_channels=A_channels, nt=nt,
                       class_weight=class_weight, rec_weight=rec_weight)
    elif model_name == 'prednet_additive':
        return PredNetAdditive(R_channels=R_channels, A_channels=A_channels, nt=nt,
                               class_weight=1, rec_weight=0)
    elif model_name == 'prednet_feedforward':
        return PredNetFF(R_channels=R_channels, A_channels=A_channels, nt=nt)


"""MODELS = {
    'prednet': PredNet(R_channels=R_channels, A_channels=A_channels, nt=nt,
                       class_weight=0.9, rec_weight=0.1),
    'prednet_additive': PredNetAdditive(R_channels=R_channels, A_channels=A_channels, nt=nt,
                                        class_weight=1, rec_weight=0, noise_type=NOISE_TYPE,
                                        noise_intensities=NOISE_INTENSITIES),
    'prednet_feedforward': PredNetFF(R_channels=R_channels, A_channels=A_channels, nt=nt, noise_type=NOISE_TYPE,
                                     noise_intensities=NOISE_INTENSITIES),
    'prednet_norec': PredNet(R_channels=R_channels, A_channels=A_channels, nt=nt,
                             class_weight=1, rec_weight=0),
    'prednet_cw': PredNet(R_channels=R_channels, A_channels=A_channels, nt=nt,
                          class_weight=0.5, rec_weight=0.5),
}
DATASETS = {'mnist_frames': MNIST_Frames, 'CK': CK, 'CKStatic': CKStatic}"""


def get_dataset_by_name(name, nt, train, transforms, half):
    """
    :param name: Name of dataset to get
    :param nt: How many timesteps (frames) to get
    :param train: Whether to get the training or validation set
    :param transforms: What transforms to apply to the image. Should be a torchvision object
    :param half: Which half of the image to get (either 'top', 'bottom', or None)
    :return: The dataset we want
    """
    if name == 'mnist_frames':
        return MNIST_Frames(nt, train, transforms)
    elif name == 'CK':
        return CK(nt, train, transforms)
    elif 'CKStatic' in name:
        # The last characters in the name string should be the number frame to get
        for i in range(len(name)):
            try:
                n_frame = int(name[i:])
                break
            except ValueError:
                pass
        else:
            raise NameError("Invalid CKStatic dataset name. "
                            "Dataset names should end with a "
                            "frame number specifying the frame to get.")
        ds = CKStatic(nt, train, transforms)
        ds.set_n_frame(n_frame)
        ds.set_half(half)
        return ds


