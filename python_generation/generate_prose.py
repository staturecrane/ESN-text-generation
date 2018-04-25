import argparse
import os
import random

from config import ALL_LETTERS, N_LETTERS
import numpy as n
from python_generation import ESN, utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help='text file to use for generation')
args = parser.parse_args()

cuda = torch.cuda.is_available()

text = utils.read_prose_lines(os.path.abspath(args.file))
corpora_length = len(text)

rnn = ESN(101, 101, r_size=100, spectral_radius=2.5)

if cuda:
    rnn = rnn.cuda()

for param in rnn.parameters():
    param.requires_grad = False

for param in rnn.W_out.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, rnn.parameters()), lr=1e-3, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9999)

def train(input_line_tensor, target_line_tensor):
    rnn.train()
    hidden  = rnn.init_hidden()
    if cuda:
        hidden = hidden.cuda()
        input_line_tensor = input_line_tensor.cuda()
        target_line_tensor = target_line_tensor.cuda()

    rnn.zero_grad()

    loss = 0
    for i in range(target_line_tensor.size()[0]):
        output, hidden = rnn(input_line_tensor[i][0], hidden)
        loss += criterion(output, target_line_tensor[i])
        
    loss.backward()
    optimizer.step()
    
    return output, loss.data.mean() / input_line_tensor.size()[0]


def sample(start_letters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", max_length=100):
    rnn.eval()
    start_letter = start_letters[random.randint(0, len(start_letters) - 1)]
    input = Variable(utils.create_input_tensor(start_letter))
    hidden = rnn.init_hidden()

    if cuda:
        hidden = hidden.cuda()

    output_program = start_letter

    for i in range(max_length):
        if cuda:
            input = input.cuda()

        output, hidden = rnn(input[0][0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == N_LETTERS - 1:
            break
        else:
            letter = ALL_LETTERS[topi]
            output_program += letter
        input = Variable(utils.create_input_tensor(letter))
    return output_program


n_epochs = 200000
print_every = 100
sample_every = 100
save_every = 1000

for epoch in tqdm(range(1, n_epochs)):
    random_idx = random.randint(0, corpora_length - 1)
    output, loss = train(*utils.random_training_set(text[random_idx], sample_length=50))
    scheduler.step()
    if epoch % print_every == 0:
        print("epoch {}: {}".format(epoch, loss))

    if epoch % sample_every == 0:
        print(sample())

    """
    if epoch % save_every == 0:
        with open('checkpoints/model.pt', 'wb') as f:
            torch.save(rnn, f)
    """
