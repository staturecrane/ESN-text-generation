import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class GRU(nn.Module):
    def __init__(self, input_size, output_size, r_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.r_size = r_size

    def init_hidden(self):
        return Variable(torch.zeros(self.r_size))


    def forward(self, input_tensor, prev_hidden):
        x_in = self.W_in.mv(input_tensor)
        x_W = self.W.mm(prev_hidden)
        update = self.tanh(x_in + x_W)
     
        return update
