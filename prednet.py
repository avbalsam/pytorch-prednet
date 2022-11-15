import torch
import torch.nn as nn
from torch.nn import functional as F
from convlstmcell import ConvLSTMCell
from torch.autograd import Variable


from debug import info


class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels, use_cuda=False, nt=5, class_weight=0.9, rec_weight=0.1):
        super(PredNet, self).__init__()
        self.classification_steps = None
        self.reconstruction_error = None

        self.time_loss_weights = torch.ones(nt, 1)
        self.layer_loss_weights = torch.FloatTensor([[1.], [1.], [1.], [1.]])
        if use_cuda:
            self.layer_loss_weights = Variable(self.layer_loss_weights.cuda())
            self.time_loss_weights = Variable(self.time_loss_weights.cuda())
        else:
            self.layer_loss_weights = Variable(self.layer_loss_weights.cpu())
            self.time_loss_weights = Variable(self.time_loss_weights.cpu())

        self.nt = nt
        self.class_weight = class_weight
        self.rec_weight = rec_weight

        self.r_channels = R_channels + (0, )  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)

        self.use_cuda = use_cuda

        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i+1],                                                                             self.r_channels[i],
                                (3, 3))
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)

        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear layer for classification
        self.linear = nn.Linear(6144, 10)

        self.flatten = nn.Flatten()

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

        self.reset_parameters()

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def forward(self, input):
        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = Variable(torch.zeros(batch_size, 2*self.a_channels[l], w, h))
            R_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], w, h))
            if self.use_cuda:
                E_seq[l] = E_seq[l].cuda()
                R_seq[l] = R_seq[l].cuda()
            else:
                E_seq[l] = E_seq[l].cpu()
                R_seq[l] = R_seq[l].cpu()
            w = w//2
            h = h//2
        time_steps = input.size(1)
        total_error = []

        classification_steps = list()
        for t in range(time_steps):
            A = input[:, t]
            if self.use_cuda:
                A = A.type(torch.cuda.FloatTensor)
            else:
                A = A.type(torch.FloatTensor)
            
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
                if l == 0:
                    frame_prediction = A_hat
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
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

        classification = sum(classification_steps) / len(classification_steps)

        loc_batch = reconstruction_error.size(0)
        rec_error = torch.mm(reconstruction_error.view(-1, self.nt), self.time_loss_weights)  # batch*n_layers x 1
        rec_error = torch.mm(rec_error.view(loc_batch, -1), self.layer_loss_weights)
        rec_error = torch.mean(rec_error)

        return rec_error, classification

    def calculate_loss(self, model_output, labels):
        rec_error, classification = model_output
        # Create a tensor of label arrays to compare with classification tensor
        label_arr = [[float(label == labels[i]) for label in range(10)] for i in range(16)]
        class_error = list()
        for c in range(len(classification)):
            if self.use_cuda:
                label_tensor = torch.cuda.FloatTensor(label_arr[c])
            else:
                label_tensor = torch.FloatTensor(label_arr[c])
            classification_loss = torch.nn.functional.cross_entropy(classification[c], label_tensor)
            class_error.append(classification_loss)

        mean_class_error = sum(class_error) / len(class_error)
        errors_total = (self.rec_weight * rec_error) + (self.class_weight * mean_class_error)

        return errors_total


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
