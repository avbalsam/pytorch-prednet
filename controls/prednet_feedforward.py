import torch
import torch.nn as nn
from torch.nn import functional as F
from convlstmcell import ConvLSTMCell
from torch.autograd import Variable


from debug import info
from prednet import PredNet


class PredNetFF(PredNet):
    def __init__(self, R_channels, A_channels, nt=5, noise_type='gaussian', noise_intensities=None):
        super().__init__(R_channels, A_channels, nt, class_weight=1, rec_weight=0, noise_type=noise_type, noise_intensities=noise_intensities)

        # Linear layer for classification
        self.linear = nn.Linear(3072, 10)

        # Use half the number of channels as regular prednet
        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

    def get_name(self):
        return f"{super().get_name()}_feedforward"

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def forward(self, input, timestep=None):
        E_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = Variable(torch.zeros(batch_size, 2 * self.a_channels[l], w, h).to(self.device))
            w = w//2
            h = h//2
        time_steps = input.size(1)
        total_error = []

        classification_steps = list()
        for t in range(time_steps):
            A = input[:, t]
            A = A.to(self.device)

            for l in range(self.n_layers):
                # conv = getattr(self, 'conv{}'.format(l))
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(A)

            # Flatten reconstruction of each layer and express as 1D vector
            flattened = self.flatten(A)
            classification = self.linear(flattened)

            mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
            # batch x n_layers
            total_error.append(mean_error)

            classification_steps.append(classification)

        reconstruction_error = torch.stack(total_error, 2) # batch x n_layers x nt
        # return torch.stack(total_error, 2), classification_steps  # batch x n_layers x nt

        if timestep is None:
            classification = sum(classification_steps) / len(classification_steps)
        else:
            classification = classification_steps[timestep]

        loc_batch = reconstruction_error.size(0)
        rec_error = torch.mm(reconstruction_error.view(-1, self.nt), self.time_loss_weights)  # batch*n_layers x 1
        rec_error = torch.mm(rec_error.view(loc_batch, -1), self.layer_loss_weights)
        rec_error = torch.mean(rec_error)

        self.rec_error = rec_error
        self.classification_steps = classification_steps

        return classification

    def get_total_error(self) -> float:
        return self.class_error


class SatLU(nn.Module):
    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)


    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
	        + ', max_val=' + str(self.upper) \
	        + inplace_str + ')'