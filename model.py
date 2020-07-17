import math
from typing import Tuple

import torch
import torch.nn as nn


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


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer, implemented as a Module.
    """

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 5000):
        """
        Constructor for a positional encoding.

        :param d_model: The Transformer's input size
        :param dropout: The dropout rate
        :param max_len: The maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Adds positional encoding to a tensor.

        :param x: A tensor
        :return: x, with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Container module with an encoder, a transformer module, and a
    decoder.
    """

    def __init__(self, ntoken: int, ninp: int, nhead: int, nhid: int,
                 nlayers: int, dropout: float = 0.5):
        """
        Constructor for a Transformer model.

        :param ntoken: The vocab size
        :param ninp: The embedding size
        :param nhead: The number of attention heads
        :param nhid: The hidden size
        :param nlayers: The number of layers
        :param dropout: The dropout rate
        """
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.ninp = ninp

        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.src_mask = None

        # Initialize Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         nlayers)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

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
        return self.decoder(output)
