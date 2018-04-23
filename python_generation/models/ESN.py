import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class ESN(nn.Module):
    def __init__(self, input_size, output_size, r_size=500, spectral_radius=0.95, sparsity=0):
        super(ESN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.r_size = r_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.random_state = np.random.RandomState(int(time.time()))

        self.input_layer = nn.Linear(input_size, r_size)
        self.hidden_layer = nn.RNNCell(r_size, r_size, bias=False)
        self.output_layer = nn.Linear(r_size, output_size)

        for param in self.hidden_layer.parameters():
            reservoir = self.random_state.rand(param.size(0), param.size(1)) - 0.5
            reservoir[self.random_state.rand(*reservoir.shape) < self.sparsity] = 0
            radius = np.max(np.abs(np.linalg.eigvals(reservoir)))
            param.data = torch.from_numpy(reservoir * (self.spectral_radius / radius)).float()

    def init_hidden(self):
        return Variable(torch.zeros(self.r_size))

    def forward(self, input_tensor, prev_hidden):
        x2h = self.input_layer(input_tensor)
        hidden = self.hidden_layer(x2h, prev_hidden)
        h2o = self.output_layer(hidden)
        return h2o, hidden
