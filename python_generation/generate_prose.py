import argparse
import os
import random

from config import ALL_LETTERS, N_LETTERS
import numpy as np
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

hidden_size = 200

rnn = ESN(101, 101, r_size=hidden_size, spectral_radius=1.0)
out = nn.Linear(hidden_size * hidden_size, 101)

if cuda:
    rnn = rnn.cuda()
    out = out.cuda()

for param in rnn.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(out.parameters(), lr=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.999)

def train(input_line_tensor, target_line_tensor):
    rnn.train()
    hidden = rnn.init_hidden()
    if cuda:
        hidden = hidden.cuda()
        input_line_tensor = input_line_tensor.cuda()
        target_line_tensor = target_line_tensor.cuda()

    rnn.zero_grad()

    states = []
    for i in range(target_line_tensor.size()[0]):
        hidden = rnn(input_line_tensor[i][0], hidden)
        states.append((hidden, target_line_tensor[i]))

    # randomize for better diversity
    idxs = list(range(target_line_tensor.size()[0]))
    random.shuffle(idxs)
    total_loss = 0
    for i in idxs:
        output = out(states[i][0].view(-1).unsqueeze(0))
        loss = criterion(output, states[i][1])

        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]

    return output, total_loss / input_line_tensor.size()[0]


def sample(start_letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ"', max_length=250):
    rnn.eval()
    out.eval()

    start_letter = start_letters[random.randint(0, len(start_letters) - 1)]
    input = Variable(utils.create_input_tensor(start_letter))
    hidden = rnn.init_hidden()

    if cuda:
        hidden = hidden.cuda()

    output_program = start_letter

    for i in range(max_length):
        if cuda:
            input = input.cuda()

        hidden = rnn(input[0][0], hidden)

        output = out(hidden.view(-1)).unsqueeze(0)
        predictions = nn.Softmax()(output).data[0]
        topi = np.argmax(predictions)

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
sample_length = 50

for epoch in tqdm(range(1, n_epochs)):
    random_idx = random.randint(0, corpora_length - 1)
    output, loss = train(*utils.random_training_set(text[random_idx], sample_length=sample_length))
    
    # scheduler.step()
    if epoch % print_every == 0:
        print("epoch {}: {}".format(epoch, loss))

    if epoch % sample_every == 0:
        print(sample(max_length=50))

    """
    if epoch % save_every == 0:
        with open('checkpoints/model.pt', 'wb') as f:
            torch.save(rnn, f)
    """
