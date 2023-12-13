import math

from torch import argmax, no_grad, save
from torch.optim import SGD
from d2l.torch import Animator, Accumulator, Timer
from tqdm import tqdm
import random
import time

from model import *
from parameters import *
from static import *


def init_model(vocab_size):
    encoder = RNNEncoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
    decoder = RNNDecoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
    model = EncoderDecoder(encoder, decoder)
    return model


def get_random():
    timestamp = int(time.time())
    random.seed(timestamp)
    return random.random()


def train(data, optimizer, BOS_TOKEN_IDX, PAD_TOKEN_IDX, device, force_teaching_ratio, net, loss, metric):
    # Initialize an accumulator for tracking training loss and word count
    metric.reset()
    # Loop over batches in the training data iterator
    for idx, batch in enumerate(data):

        print(f'Training batch Num. {idx}...')

        # Zero the gradients to prepare for a new batch
        optimizer.zero_grad()
        # Move input and target tensors to the specified device
        X, Y = [x.to(device) for x in batch]  # X.shape: [num_steps, batch_size]
        # Create a beginning-of-sequence (BOS) tensor for the target
        bos = tensor([BOS_TOKEN_IDX] * Y.shape[0], device=device).reshape((-1, 1))

        if force_teaching_ratio >= get_random():
            # print('Turn on teacher forcing.')
            # Concatenate BOS tensor with the target sequence (force teaching)
            dec_input = cat([bos, Y[:, :-1]], 1)
            # Forward pass through the model
            Y_hat, _ = net(X, dec_input)
        else:
            # print('Turn off teacher forcing.')
            # No force teaching
            dec_input = bos
            # Initialize an empty tensor to store the decoder outputs
            outputs = []
            # Loop through the decoder time steps
            for idx in range(Y.shape[1]):
                Y_hat, _ = net(X, dec_input)
                dec_input = argmax(Y_hat, dim=-1)  # Use the predicted token as input for the next step
                outputs.append(Y_hat)
            # Concatenate the decoder outputs along the time dimension
            Y_hat = cat(outputs, dim=1)

        # Flatten the target tensor and the model's output
        Y = Y.view(-1)
        output_dim = Y_hat.shape[-1]
        Y_hat = Y_hat.contiguous().view(-1, output_dim)  # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

        # Calculate the loss using MaskedSoftmaxCELoss
        model_loss = loss(Y_hat, Y, PAD_TOKEN_IDX)
        # Backward pass to compute the gradients
        model_loss.sum().backward()
        # Count the number of non-padding tokens
        num_tokens = (Y != 0).sum()
        # Update the model parameters using the optimizer
        optimizer.step()
        # Accumulate training loss and word count
        with no_grad():
            metric.add(model_loss.sum().item(), num_tokens.item())

    return metric


def evaluate(data, BOS_TOKEN_IDX, PAD_TOKEN_IDX, device, net, loss, metric):
    # Initialize an accumulator for tracking training loss and word count
    metric.reset()
    # Loop over batches in the training data iterator
    for i, batch in enumerate(data):
        print(f'Evaluating batch Num. {i}...')
        # Move input and target tensors to the specified device
        X, Y = [x.to(device) for x in batch]  # X.shape: [num_steps, batch_size]
        # Create a beginning-of-sequence (BOS) tensor for the target
        bos = tensor([BOS_TOKEN_IDX] * Y.shape[0], device=device).reshape((-1, 1))
        # No force teaching
        dec_input = bos
        # Initialize an empty tensor to store the decoder outputs
        outputs = []
        # Loop through the decoder time steps
        for i in range(Y.shape[1]):
            Y_hat, _ = net(X, dec_input)
            dec_input = argmax(Y_hat, dim=-1)  # Use the predicted token as input for the next step
            outputs.append(Y_hat)
        # Concatenate the decoder outputs along the time dimension
        Y_hat = cat(outputs, dim=1)

        # Flatten the target tensor and the model's output
        Y = Y.view(-1)
        output_dim = Y_hat.shape[-1]
        Y_hat = Y_hat.contiguous().view(-1, output_dim)  # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

        # Calculate the loss using MaskedSoftmaxCELoss
        model_loss = loss(Y_hat, Y, PAD_TOKEN_IDX)
        # Backward pass to compute the gradients
        model_loss.sum().backward()
        # Count the number of non-padding tokens
        num_tokens = (Y != 0).sum()
        # Accumulate training loss and word count
        with no_grad():
            metric.add(model_loss.sum().item(), num_tokens.item())

    return metric


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_eval(train_data, valid_data, vocab, device, force_teaching_ratio):
    print('Start training...')
    model = init_model(len(vocab))
    # Put data_iter in a list to iterate repeatedly
    train_data_list = [item for item in train_data]
    valid_data_list = [item for item in valid_data]
    print('Training dataset length: ', len(train_data_list))
    print('Validating dataset length: ', len(valid_data_list))
    # Move the model to the specified device (GPU or CPU)
    model.to(device)
    # Define the optimizer with stochastic gradient descent (SGD) and the given learning rate
    optimizer = SGD(model.parameters(), lr=lr)

    # Define the loss function
    loss = MaskedSoftmaxCELoss()
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    # Initialize an accumulator for tracking training loss and word count
    train_metric = Accumulator(2)  # 训练损失总和，单词数量
    valid_metric = Accumulator(2)  # 训练损失总和，单词数量

    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch} / {num_epochs}: ')
        timer = Timer()
        # Set the model in training mode
        model.train()
        print('Training...')
        train_metric = train(train_data_list, optimizer, vocab[BOS_TOKEN], vocab[PAD_TOKEN],
                             device, force_teaching_ratio, model, loss, train_metric)
        # if (epoch + 1) % 10 == 0:
        animator.add(epoch + 1, (train_metric[0] / train_metric[1],))
        # print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train loss {train_metric[0] / train_metric[1]:.3f}, {train_metric[1] / timer.stop():.1f} '
              f'tokens/sec on {str(device)}')
        print('Validating...')
        # Set the model in evaluating mode
        model.eval()
        valid_metric = evaluate(valid_data_list, vocab[BOS_TOKEN], vocab[PAD_TOKEN],
                                device, model, loss, valid_metric)
        print(f'Valid loss {valid_metric[0] / valid_metric[1]:.3f}, {valid_metric[1] / timer.stop():.1f} '
              f'tokens/sec on {str(device)}')

    # Save the trained model
    save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
