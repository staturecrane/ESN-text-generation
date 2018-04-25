import argparse
import os
import random

from config import ALL_LETTERS, N_LETTERS
import numpy as n
from python_generation import ESN, utils
import torch
from torch.autograd import Variable
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help='text file to use for generation')
args = parser.parse_args()

programs = utils.read_lines(os.path.abspath(args.file))
print(len(programs))

rnn = ESN(101, 101)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)


def train(input_line_tensor, target_line_tensor):

    hidden  = rnn.init_hidden()

    rnn.zero_grad()

    loss = 0
    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()

    optimizer.step()

    return output, loss.data[0] / input_line_tensor.size()[0]


def sample(start_letters="#idaue", max_length=200):
    start_letter = start_letters[random.randint(1, len(start_letters) - 1)]
    input = Variable(utils.create_input_tensor(start_letter))
    hidden = rnn.init_hidden()

    output_program = start_letter

    for i in range(max_length):
        output, hidden = rnn(input[0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == N_LETTERS - 1:
            break
        else:
            letter = ALL_LETTERS[topi]
            output_program += letter
        input = Variable(utils.create_input_tensor(letter))
    return output_program


n_epochs = 10000
print_every = 100
sample_every = 500
save_every = 1000

for epoch in range(1, n_epochs):
    output, loss = train(*utils.random_training_set(programs, sample_length=100))

    if epoch % print_every == 0:
        print("epoch {}: {}".format(epoch, loss))

    if epoch % sample_every == 0:
        print(sample())

    """
    if epoch % save_every == 0:
        with open('checkpoints/model.pt', 'wb') as f:
            torch.save(rnn, f)
    """
