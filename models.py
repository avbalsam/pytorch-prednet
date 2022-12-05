import torch

from controls.prednet_additive import PredNetAdditive
from controls.prednet_feedforward import PredNetFF
from mnist_data_prednet import MNIST_Frames
from prednet import PredNet

A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
nt = 10  # Number of timesteps
NOISE_TYPE = 'gaussian'
NOISE_INTENSITIES = [0.0, 0.25, 0.5]


# NOISE_INTENSITIES = [0.0]


def get_model_by_name(model_name, nt, class_weight, rec_weight, noise_type, noise_intensities):
    if model_name == 'prednet':
        return PredNet(R_channels=R_channels, A_channels=A_channels, nt=nt,
                       class_weight=class_weight, rec_weight=rec_weight, noise_type=noise_type,
                       noise_intensities=noise_intensities)
    elif model_name == 'prednet_additive':
        return PredNetAdditive(R_channels=R_channels, A_channels=A_channels, nt=nt,
                               class_weight=1, rec_weight=0, noise_type=noise_type, noise_intensities=noise_intensities)
    elif model_name == 'prednet_feedforward':
        return PredNetFF(R_channels=R_channels, A_channels=A_channels, nt=nt, noise_type=noise_type,
                         noise_intensities=noise_intensities)


MODELS = {
    'prednet': PredNet(R_channels=R_channels, A_channels=A_channels, nt=nt,
                       class_weight=0.1, rec_weight=0.9, noise_type=NOISE_TYPE, noise_intensities=NOISE_INTENSITIES),
    'prednet_additive': PredNetAdditive(R_channels=R_channels, A_channels=A_channels, nt=nt,
                                        class_weight=1, rec_weight=0, noise_type=NOISE_TYPE,
                                        noise_intensities=NOISE_INTENSITIES),
    'prednet_feedforward': PredNetFF(R_channels=R_channels, A_channels=A_channels, nt=nt, noise_type=NOISE_TYPE,
                                     noise_intensities=NOISE_INTENSITIES),
    'prednet_norec': PredNet(R_channels=R_channels, A_channels=A_channels, nt=nt,
                             class_weight=1, rec_weight=0, noise_type=NOISE_TYPE, noise_intensities=NOISE_INTENSITIES),
    'prednet_cw': PredNet(R_channels=R_channels, A_channels=A_channels, nt=nt,
                          class_weight=0.5, rec_weight=0.5, noise_type=NOISE_TYPE, noise_intensities=NOISE_INTENSITIES),
}
DATASETS = {'mnist_frames': MNIST_Frames}
