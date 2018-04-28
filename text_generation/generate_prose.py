import argparse
import os
import random

from config import ALL_LETTERS, N_LETTERS
import numpy as np
from text_generation import ESN, utils
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

hidden_size = 500

rnn = ESN(101, 101, r_size=hidden_size, spectral_radius=2.0)
out = nn.Linear(hidden_size * hidden_size, 101)

if cuda:
    rnn = rnn.cuda()
    out = out.cuda()

# uncomment if you wish to freeze the input and hidden weights 
# (this is the original ESN approach)
for param in rnn.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(out.parameters(), lr=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9999)

def train(input_line_tensor, target_line_tensor):
    rnn.train()
    hidden = rnn.init_hidden()
    if cuda:
        hidden = hidden.cuda()
        input_line_tensor = input_line_tensor.cuda()
        target_line_tensor = target_line_tensor.cuda()

    rnn.zero_grad()

    loss = 0
    for i in range(target_line_tensor.size()[0]):
        hidden = rnn(input_line_tensor[i][0], hidden)

        hidden_data = hidden.view(-1).unsqueeze(0).data
        output = out(Variable(hidden_data))
        loss += criterion(output, target_line_tensor[i])

    loss.backward()
    optimizer.step()

    return output, loss.data[0] / input_line_tensor.size()[0]


def sample(input_tensor, max_length=250, temperature=0.8):
    rnn.eval()
    out.eval()

    hidden = rnn.init_hidden()

    if input_tensor.size()[0] < 3:
        length = input_tensor.size()[0]
    else:
        length = 3

    if cuda:
        hidden = hidden.cuda()
        input_tensor = input_tensor.cuda()

    output_program = ''
    for i in range(length):
        hidden = rnn(input_tensor[i][0], hidden)
        topi = np.argmax(input_tensor[i][0].data)
        letter = ALL_LETTERS[topi]

        output_program += letter

    input = Variable(utils.create_input_tensor(letter))
    for i in range(max_length):
        if cuda:
            input = input.cuda()

        hidden = rnn(input[0][0], hidden)

        output = out(hidden.view(-1)).unsqueeze(0)

        predictions = nn.Softmax()(output).data[0]
        preds = np.asarray(predictions).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        topi = np.argmax(probas)

        if topi == N_LETTERS - 1:
            break
        else:
            letter = ALL_LETTERS[topi]
            output_program += letter
        input = Variable(utils.create_input_tensor(letter))
    return output_program


n_epochs = 2000000
print_every = 100
sample_every = 100
save_every = 1000
sample_length = 300

for epoch in tqdm(range(1, n_epochs)):
    random_idx = random.randint(0, corpora_length - 1)
    output, loss = train(
        *utils.random_training_set(text[random_idx], sample_length=sample_length, 
        volatile=True
    ))

    # scheduler.step()
    if epoch % print_every == 0:
        print("epoch {}: {}".format(epoch, loss))

    if epoch % sample_every == 0:
        random_sample_idx = random.randint(0, corpora_length - 1)
        input_line_tensor, _ = utils.random_training_set(
            text[random_idx], sample_length=sample_length, volatile=True
        )
        print(sample(input_line_tensor, max_length=sample_length))

    """
    if epoch % save_every == 0:
        with open('checkpoints/model.pt', 'wb') as f:
            torch.save(rnn, f)
    """
