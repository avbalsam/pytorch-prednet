import torch

from controls.prednet_additive import PredNetAdditive
from controls.prednet_feedforward import PredNetFF
from mnist_data_prednet import MNIST_Frames
from prednet import PredNet

A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
nt = 5  # Number of timesteps

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

MODELS = {
    'prednet': PredNet(R_channels=R_channels, A_channels=A_channels, nt=nt,
                       class_weight=0.1, rec_weight=0.9, device=device),
    'prednet_additive': PredNetAdditive(R_channels=R_channels, A_channels=A_channels, nt=nt,
                                        class_weight=1, rec_weight=0, device=device),
    'prednet_feedforward': PredNetFF(R_channels=R_channels, A_channels=A_channels, nt=nt, device=device),
    'prednet_norec': PredNet(R_channels=R_channels, A_channels=A_channels, nt=nt,
                             class_weight=1, rec_weight=0, device=device)}
DATASETS = {'mnist_frames': MNIST_Frames}
