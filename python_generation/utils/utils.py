import unicodedata
import random
import time
import math

from python_generation.config import ALL_LETTERS, N_LETTERS, SAMPLE_LENGTH
import torch
from torch.autograd import Variable


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


def read_lines(filename):
    program = open(filename).read().split('--------')
    while '' in program:
        program.remove('')
    return [unicode_to_ascii(line) for line in program]


def read_prose_lines(filename):
    text = open(filename).read().split('.')
    only_long = filter(lambda line: len(line) > 10, text)
    stripped = [line.strip() for line in only_long]
    return [unicode_to_ascii(line) for line in stripped]


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def create_input_tensor(line):
    length = SAMPLE_LENGTH if len(line) > SAMPLE_LENGTH else len(line)
    tensor = torch.zeros(length, 1, N_LETTERS)
    for li in range(length):
        letter = line[li]
        tensor[li][0][ALL_LETTERS.find(letter)] = 1
    return tensor


def create_target_tensor(line):
    length = SAMPLE_LENGTH if len(line) > SAMPLE_LENGTH else len(line)
    letter_indexes = [ALL_LETTERS.find(line[li]) for li in range(1,
        length)]
    letter_indexes.append(N_LETTERS - 1)
    return torch.LongTensor(letter_indexes)


def random_training_set(line, sample_length=500):
    length = sample_length if len(line) > sample_length else len(line)
    if length > sample_length:
        random_idx = torch.randint(0, length - sample_length - 1)
        line = line[random_idx : random_idx + sample_length + 1]
    input_line_tensor = Variable(create_input_tensor(line))
    target_line_tensor = Variable(create_target_tensor(line))

    return input_line_tensor, target_line_tensor
