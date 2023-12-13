import random

import numpy as np
import torch
from d2l.torch import tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from static import *
from d2l import torch as d2l


def tokenize(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def build_vocab(data_file, min_freq, vocab_file):
    root_file = './data/'
    text = open(root_file + data_file, 'r', encoding='utf-8').read()
    src_tokenized, trg_tokenized = tokenize(text)
    vocab = d2l.Vocab(src_tokenized + trg_tokenized, min_freq=min_freq,
                      reserved_tokens=[UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    print(f'Vocabulary size: {len(vocab)}')
    torch.save(vocab, root_file + vocab_file)
    print(f'Vocabulary saved in {vocab_file}')


# build_vocab('dialogues_vocab.txt', 5, 'vocab.pth')

vocab = torch.load('./data/vocab.pth')
print(len(vocab))
print(vocab.token_freqs[:10])
