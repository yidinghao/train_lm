import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, rnn_type: str, ntoken: int, ninp: int, nhid: int,
                 nlayers: int, dropout: float = 0.5,
                 tie_weights: bool = False):
        """
        Constructor for an RNNModel.

        :param rnn_type: The RNN architecture: LSTM, GRU, RNN_TANH, or
            RNN_RELU
        :param ntoken: The vocab size
        :param ninp: The embedding size
        :param nhid: The hidden size
        :param nlayers: The number of RNN layers
        :param dropout: The dropout weight
        :param tie_weights: Whether to tie the input and output
            embeddings
        """
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)

        # Initialize the right type of RNN
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(ninp, nhid, nlayers, dropout=dropout)
        elif rnn_type == "RNN_TANH":
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity="tanh",
                              dropout=dropout)
        elif rnn_type == "RNN_RELU":
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity="relu",
                              dropout=dropout)
        else:
            raise ValueError("The value of --model must be LSTM, GRU,"
                             "RNN_TANH, or RNN_RELU, not {}.".format(rnn_type))

        # Optionally tie weights as in Press and Wolf (2016). See the
        # paper for more details: https://arxiv.org/abs/1608.05859
        if tie_weights and nhid == ninp:
            self.decoder.weight = self.encoder.weight
        elif tie_weights:
            raise ValueError("When using --tied, --nhid must be equal to "
                             "--emsize. Currently, --nhid is {} and --emsize"
                             "is {}.".format(nhid, ninp))

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param input: An input batch
        :param hidden: The hidden state from the previous batch
        :return: The output and the hidden state
        """
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return self.decoder(output), hidden

    def init_hidden(self, batch_size: int):
        """
        Constructs a zero vector that serves as the initial hidden state
        of the RNN.

        :param batch_size: The batch size
        :return: The initial hidden state
        """
        weight = next(self.parameters())
        if isinstance(self.rnn, nn.LSTM):
            return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers, batch_size, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhid)


# Temporarily leave PositionalEncoding module here. Will be moved somewhere
# else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of
    the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and
        cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
            -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module,
    and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError(
                'TransformerEncoder module does not exist in PyTorch 1.1 or '
                'lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(
                    device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
