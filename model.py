from torch import tensor
from torch.nn import CrossEntropyLoss
from typing import Tuple
from d2l.torch import tensor
from torch import cat, nn, Tensor, long


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X) -> Tensor:
        raise NotImplementedError


class RNNEncoder(Encoder):
    """用RNN实现编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(RNNEncoder, self).__init__(**kwargs)
        # 嵌入层 获取输入序列中每个单词的嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 基于GRU实现
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X) -> Tensor:
        """
        Args:
            X.shape: [batch_size, num_steps]

        Returns:
        """
        X = tensor(X, dtype=long)
        X = self.embedding(X)  # shape: [batch_size, num_steps, embed_size]
        X = X.permute((1, 0, 2))  # shape: [num_steps, batch_size, embed_size]
        output, state = self.rnn(X)
        return output, state


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs):
        raise NotImplementedError

    def forward(self, X, state) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class RNNDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=.0, **kwargs):
        super(RNNDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # embed_size + num_hiddens 为了处理拼接后的维度，见forward函数中的注释
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, batch_first=False, dropout=dropout)
        # 将隐状态转换为词典大小维度
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs):
        return enc_outputs[1]  # 得到编码器输出中的state

    def forward(self, X, state) -> Tuple[Tensor, Tensor]:
        X = tensor(X, dtype=long)
        X = self.embedding(X).permute((1, 0, 2))  # (num_steps, batch_size, embed_size)

        # 将最顶层的上下文向量广播成与X相同的时间步，其他维度上只复制1次(保持不变)
        # 形状 (num_layers, batch_size, num_hiddens ) => (num_steps, batch_size, num_hiddens)
        context = state[-1].repeat((X.shape[0], 1, 1))
        # 为了每个解码时间步都能看到上下文，拼接context与X
        # (num_steps, batch_size, embed_size) + (num_steps, batch_size, num_hiddens)
        #                           => (num_steps, batch_size, embed_size + num_hiddens)
        concat_context = cat((X, context), 2)

        output, state = self.rnn(concat_context, state)
        output = self.dense(output).permute((1, 0, 2))  # (batch_size, num_steps, vocab_size)

        return output, state


class MaskedSoftmaxCELoss(CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, padding_value):
        self.reduction = 'none'
        label = tensor(label, dtype=long)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred, label)
        label_valid = label != padding_value
        # with no_grad():
        #    num_tokens = label_valid.sum().item()

        # weighted_loss = (unweighted_loss * label_valid).sum() / float(num_tokens)
        weighted_loss = (unweighted_loss * label_valid).sum()
        return weighted_loss


class EncoderDecoder(nn.Module):
    """合并编码器和解码器"""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X) -> Tensor:
        enc_outputs = self.encoder(X=enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)
