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

        input_weights = self.random_state.rand(r_size, input_size) * 2 - 1

        self.W_in = nn.Parameter(torch.from_numpy(input_weights.astype(np.float32)))
        self.W_out = nn.Linear(r_size * r_size, output_size, bias=True)
        torch.nn.init.xavier_normal(self.W_out.weight)
        self.W = nn.Parameter(self.init_reservoir())

        self.tanh = nn.Tanh()

    def init_reservoir(self):
        reservoir = self.random_state.rand(self.r_size, self.r_size) - 0.5
        reservoir[self.random_state.rand(*reservoir.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(reservoir)))
        return torch.from_numpy((reservoir * (self.spectral_radius / radius)).astype(np.float32))
    

    def init_hidden(self):
        reservoir = np.zeros((self.r_size, self.r_size))
        return Variable(torch.from_numpy(reservoir).float())


    def forward(self, input_tensor, prev_hidden):
        x_in = self.W_in.mv(input_tensor)
        x_W = self.W.mm(prev_hidden)
        # noise = torch.from_numpy(self.noise * self.random_state.rand(self.r_size) - 0.5).float()
        activated = self.tanh(x_in + x_W) # + (0.001 * Variable(noise, requires_grad=False)))
        prediction = self.W_out(activated.view(-1).unsqueeze(0))

        return prediction, activated
