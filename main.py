import torch
from torch import load

from prepare_data import load_dataset
from train_eval import train_eval
from run import run
from parameters import *


class Seq2seq:
    def __init__(self):
        self.vocab = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.vocab = load(root_path + vocab_path)
        print(f'Vocabulary size: {len(self.vocab)}')

    def train(self, train_data_file, valid_data_file):
        print('Loading training dataset...')
        train_data_loader = load_dataset(self.vocab, train_data_file)
        print('Loading validating dataset...')
        valid_data_loader = load_dataset(self.vocab, valid_data_file)
        train_eval(train_data_loader, valid_data_loader, self.vocab, self.device, force_teaching_ratio=0.5)

    def run(self, text):
        return run(self.vocab, text, self.device)


if __name__ == '__main__':
    # process_dialog(input_file_name='dialogues_train.txt', output_file_name='dialogues_train_processed.txt')
    # process_dialog(input_file_name='dialogues_validation.txt', output_file_name='dialogues_valid_processed.txt')
    # build_vocab(data_file='dialogues_vocab.txt', min_freq=min_freq, vocab_file='vocab.pth')
    seq2seq = Seq2seq()
    seq2seq.train('dialogues_train_processed.txt', 'dialogues_valid_processed.txt')
    input_str = 'Christmas is coming. They must be popular again this season.'
    print('Input: ', input_str)
    print('Output: ', seq2seq.run(input_str))
