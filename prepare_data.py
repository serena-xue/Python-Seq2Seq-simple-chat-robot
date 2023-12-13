import random

import numpy as np
from d2l.torch import tensor
from torch import load
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from static import *
from parameters import *


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


def encode(sentences, vocab):
    sentences = [vocab[sentence] for sentence in sentences]
    sentences = [[vocab[BOS_TOKEN]] + sentence + [vocab[EOS_TOKEN]] for sentence in sentences]
    return sentences


# def build_array(lines, vocab, max_len=None):
#     if not max_len:
#         max_len = max(len(x) for x in lines)
#
#     """将机器翻译的文本序列转换成小批量"""
#     lines = [vocab[line] for line in lines]
#     lines = [[vocab[BOS_TOKEN]] + line + [vocab[EOS_TOKEN]] for line in lines]
#     return np.array([truncate_pad(line, max_len, vocab[PAD_TOKEN]) for line in lines])


def truncate_pad_line(line, max_len):
    if len(line) > max_len:
        return line[:max_len]
    return tensor(line + [PAD_TOKEN_IDX] * (max_len - len(line)))


def truncate_pad(list1, list2):
    max_length_1 = max(len(sublist) for sublist in list1)
    max_length_2 = max(len(sublist) for sublist in list2)
    max_len = max(max_length_1, max_length_2, seq_len)
    result1 = [truncate_pad_line(line, max_len) for line in list1]
    result2 = [truncate_pad_line(line, max_len) for line in list2]
    return result1, result2


class NMTDataset(Dataset):
    def __init__(self, data):
        self.data = np.array(data, dtype=list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @staticmethod
    def collate_fn(examples):
        # src = [tensor(ex[0]) for ex in examples]
        src = [ex[0] for ex in examples]
        # trg = [tensor(ex[1]) for ex in examples]
        trg = [ex[1] for ex in examples]
        src, trg = truncate_pad(src, trg)
        # 这里只是转换为Tensor向量
        src = pad_sequence(src)
        trg = pad_sequence(trg)
        return src, trg

    @staticmethod
    def batch_sampler(examples, batch_size):
        indices = [(i, len(s[0])) for i, s in enumerate(list(examples))]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))
        pooled_indices = [x[0] for x in pooled_indices]
        # yield indices for current batch
        for i in range(0, len(pooled_indices), batch_size):
            yield pooled_indices[i:i + batch_size]


def load_dataset(vocab, data_path):
    global PAD_TOKEN_IDX
    PAD_TOKEN_IDX = vocab[PAD_TOKEN]
    text = open(root_path + data_path, 'r', encoding='utf-8').read()
    src_tokenized, trg_tokenized = tokenize(text)
    source = encode(src_tokenized, vocab)
    target = encode(trg_tokenized, vocab)
    dataset = NMTDataset([(src, tgt) for src, tgt in zip(source, target)])
    # 数据加载器
    data_loader = DataLoader(dataset, collate_fn=dataset.collate_fn,
                             batch_sampler=dataset.batch_sampler(dataset, batch_size))
    # 返回加载器和词表
    return data_loader


# data_loader, vocab = load_dataset('./data/dialogues_processed.txt')
