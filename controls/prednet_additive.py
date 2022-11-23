import torch
import torch.nn as nn
from torch.nn import functional as F
from convlstmcell import ConvLSTMCell
from torch.autograd import Variable


from debug import info
from prednet import PredNet


class PredNetAdditive(PredNet):
    """
    Adds errors between layers instead of subtracting them, and does not take rec error into account.
    Otherwise exactly the same as regular Prednet.
    """
    def __init__(self, R_channels, A_channels, nt=5, class_weight=1, rec_weight=0):
        super().__init__(R_channels, A_channels, nt, class_weight=class_weight, rec_weight=rec_weight)

    def get_name(self):
        return f"{super().get_name()}_additive"

    def forward(self, input, timestep=None):
        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = Variable(torch.zeros(batch_size, 2 * self.a_channels[l], w, h).to(self.device))
            R_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], w, h).to(self.device))
            w = w//2
            h = h//2
        time_steps = input.size(1)
        total_error = []

        classification_steps = list()
        for t in range(time_steps):
            A = input[:, t]
            A = A.to(self.device)

            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                if t == 0:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = H_seq[l]
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    tmp = torch.cat((E, self.upsample(R_seq[l+1])), 1)
                    R, hx = cell(tmp, hx)
                R_seq[l] = R
                H_seq[l] = hx

            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))

                A_hat = conv(R_seq[l])

                # These two lines are the only significant difference between this control model and the regular prednet
                pos = F.relu(A_hat + A)
                neg = F.relu(-A_hat - A)

                E = torch.cat([pos, neg],1)
                E_seq[l] = E

                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E)

            # Flatten reconstruction of each layer and express as 1D vector
            flattened = self.flatten(E)
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

        return rec_error, classification
