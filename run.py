import torch
from torch import load, no_grad, argmax

from model import *
from static import *
from parameters import *


def process_input(vocab, device, string):

    def no_space(char, prev_char):
        return char in set(',.!?:";') and prev_char != ' '

    # Replace unbroken spaces with spaces and convert all to lowercase
    string = string.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Inserting spaces between words and punctuation marks
    string = ''.join([' ' + char if i > 0 and no_space(char, string[i - 1]) else char
              for i, char in enumerate(string)])

    # Move input and target tensors to the specified device
    str_tokens = [vocab[token] for token in string.split()]
    str_tensor = tensor(str_tokens).reshape((-1, 1)).to(device)
    return str_tokens, str_tensor


def load_model(vocab_size,device):
    # Define the encoder and decoder
    encoder = RNNEncoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
    decoder = RNNDecoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
    # Build the Encoder-Decoder model
    model = EncoderDecoder(encoder, decoder).to(device)
    model.load_state_dict(load(model_path, map_location=torch.device(device)))  # cpu
    # model.load_state_dict(load(model_path))  # cuda
    model.eval()

    return model


def run(vocab, input_str, device):
    net = load_model(len(vocab), device)
    input_tokens, input_tensor = process_input(vocab, device, input_str)
    # Create a beginning-of-sequence (BOS) tensor for the target
    bos = tensor([vocab[BOS_TOKEN]] * len(input_tokens), device=device).reshape((-1, 1))
    dec_input = cat([bos, input_tensor[:, :-1]], 1)
    with no_grad():
        Y_hat, _ = net(input_tensor, dec_input)
    outputs = argmax(Y_hat, dim=-1)
    output_list = [vocab.idx_to_token[token.item()] for token in outputs]
    return ' '.join(output_list)
