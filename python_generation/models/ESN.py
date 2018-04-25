import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class ESN(nn.Module):
    def __init__(self, input_size, output_size, r_size=200, spectral_radius=0.95, sparsity=0):
        super(ESN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.r_size = r_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = 0.0001
        self.random_state = np.random.RandomState(int(time.time()))

        input_weights = self.random_state.rand(input_size, r_size) * 2 - 1
        self.input_layer = nn.Linear(input_size, r_size)
        for param in self.input_layer.parameters():
            try:
                param.data = torch.from_numpy(self.random_state.rand(param.size(0), param.size(1)) * 2 - 1).float()
            except:
                param.data = torch.from_numpy(self.random_state.rand(param.size(0)) * 2 - 1).float()

        self.reservoir = nn.Parameter(self.init_reservoir())
        self.fc = nn.Sequential(
            nn.Linear(self.r_size * self.r_size, self.output_size),
        )
        for param in self.fc.parameters():
            try:
                param.data = torch.from_numpy(self.random_state.rand(param.size(0), param.size(1)) * 2 - 1).float()
            except:
                param.data = torch.from_numpy(self.random_state.rand(param.size(0)) * 2 - 1).float()
        self.tanh = nn.Tanh()


    def init_reservoir(self):
        reservoir = self.random_state.rand(1, self.r_size, self.r_size) - 0.5
        reservoir[self.random_state.rand(*reservoir.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(reservoir)))
        return torch.from_numpy(reservoir * (self.spectral_radius / radius)).float()
    

    def init_hidden(self):
        reservoir = np.zeros((1, self.r_size, self.r_size))
        return Variable(torch.from_numpy(reservoir).float())


    def forward(self, input_tensor, prev_hidden):
        x2h = self.input_layer(input_tensor)
        hidden = self.reservoir.bmm(prev_hidden)
        noise = torch.from_numpy(self.noise * self.random_state.rand(self.r_size) - 0.5).float().cuda()
        activated = self.tanh(x2h + hidden + (0.001 * Variable(noise, requires_grad=False)))
        prediction = self.fc(activated.view(1, self.r_size * self.r_size))
        return prediction, activated
