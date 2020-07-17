import math
import time
from typing import Iterable, Union

import torch
import torchtext.data as tt
from torch import nn
from torch import optim
from torchtext.datasets import WikiText2

from model import RNNModel, TransformerModel

HiddenState = Union[torch.Tensor, Iterable[torch.Tensor]]
Model = Union[RNNModel, TransformerModel]


def repackage_hidden(h: HiddenState) -> HiddenState:
    """
    Wraps hidden states in new Tensors, to detach them from their
    history.

    :param h: An individual hidden state vector, or a list/tuple of such
        vectors.

    :return: The detached vectors
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def perplexity(avg_loss: float) -> Union[float, str]:
    return math.exp(avg_loss) if avg_loss > 0. else "N/A"


def train(model: Model, train_iter: tt.Iterator, epoch: int,
          criterion: nn.Module, optimizer: optim.Optimizer, clip: float,
          log_interval: int):
    """
    Trains a model for one epoch.

    :param model: A model
    :param train_iter: The training iterator
    :param epoch: The epoch number
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param clip: The amount of gradient clipping
    :param log_interval: The frequency of logging results
    :return:
    """
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(train_iter.dataset.fields["text"].vocab)
    if isinstance(model, RNNModel):
        hidden = model.init_hidden(train_iter.batch_size)

    # Training loop
    n_batches = len(train_iter)
    n_preds = 0
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        if isinstance(model, TransformerModel):
            output = model(batch.text)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(batch.text, hidden)

        loss = criterion(output.view(-1, ntokens), batch.target.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        n_preds += batch.target.numel()

        # Log results
        if (i + 1) % log_interval == 0:
            cur_loss = total_loss / n_preds
            elapsed = time.time() - start_time
            prev_i = i - log_interval + 2
            print("Epoch {}, Batch {:4}-{:4}/{} ({:5.2f} s)\tPPL: {:5.2f}"
                  "".format(epoch, prev_i, i + 1, n_batches, elapsed,
                            perplexity(cur_loss)))
            total_loss = 0
            n_preds = 0
            start_time = time.time()


def evaluate(model: Model, eval_iter: tt.Iterator, criterion: nn.Module) -> \
        float:
    """
    Computes the loss on a testing or validation set.

    :param model: The model
    :param eval_iter: The iterator
    :param criterion: The loss function
    :return: The loss
    """
    model.eval()
    total_loss = 0.
    ntokens = len(eval_iter.dataset.fields["text"].vocab)
    if isinstance(model, RNNModel):
        hidden = model.init_hidden(eval_iter.batch_size)

    with torch.no_grad():
        n_preds = 0
        for batch in eval_iter:
            if isinstance(model, TransformerModel):
                output = model(batch.text)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(batch.text, hidden)

            loss = criterion(output.view(-1, ntokens), batch.target.view(-1))
            total_loss += loss.item()
            n_preds += batch.target.numel()

    return total_loss / n_preds


def run_experiment(batch_size: int = 20, bptt: int = 35, clip: float = .25,
                   cuda: bool = False, dropout: float = .2, emsize: int = 200,
                   epochs: int = 40, log_interval: int = 200, lr: float = .2,
                   model: str = "LSTM", nhead: int = 2, nhid: int = 200,
                   nlayers: int = 2, save: str = "model.pt", tied: bool =
                   False):
    """
    Trains a language model.
    
    :param batch_size: The batch size
    :param bptt: The maximum sequence length
    :param clip: Gradient clipping
    :param cuda: Whether to use CUDA
    :param dropout: The dropout rate
    :param emsize: The embedding size
    :param epochs: The maximum number of epochs

    :param lr: The initial learning rate for Adam
    :param model: The model architecture
    :param nhead: The number of attention heads (for Transformer)
    :param nhid: The hidden size
    :param nlayers: The number of RNN/Transformer layers
    :param save: The path to save the model to
    :param tied: Whether to tie weights
    :return: None
    """
    device = torch.device("cuda" if cuda else "cpu")

    # Load data
    text_field = tt.Field()
    train_data, valid_data, test_data = WikiText2.splits(text_field)
    text_field.build_vocab(train_data)

    kwargs = dict(device=device, repeat=False)
    train_iter = tt.BPTTIterator(train_data, batch_size, bptt,
                                 **kwargs)
    val_iter = tt.BPTTIterator(valid_data, 10, bptt, **kwargs)
    test_iter = tt.BPTTIterator(test_data, 10, bptt, **kwargs)

    # Build the model
    ntokens = len(text_field.vocab)
    if model == "Transformer":
        model = TransformerModel(ntokens, emsize, nhead, nhid,
                                 nlayers, dropout).to(device)
    else:
        model = RNNModel(model, ntokens, emsize, nhid,
                         nlayers, dropout, tied).to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")

    # Epoch loop
    best_val_loss = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Train and validate
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train(model, train_iter, epoch, criterion, optimizer, clip,
              log_interval)
        val_loss = evaluate(model, val_iter, criterion)

        # Report validation loss
        epoch_time = time.time() - epoch_start_time
        print("_" * 89)
        print("| End of Epoch {:3d} ({:5.2f} s)\tValidation PPL: {:8.2f}"
              "".format(epoch, epoch_time, perplexity(val_loss)))
        print("‾" * 89)

        # Save the model if the validation loss is the best we've seen so far
        if not best_val_loss or val_loss < best_val_loss:
            with open(save, "wb") as f:
                torch.save(model, f)
            best_val_loss = val_loss

    # Load the best model
    with open(save, "rb") as f:
        model = torch.load(f)
    if isinstance(model, RNNModel):
        model.rnn.flatten_parameters()

    # Run on test data
    test_loss = evaluate(model, test_iter, criterion)
    print("_" * 89)
    print("| End of Training\tTest PPL: {:8.2f}".format(perplexity(test_loss)))
    print("‾" * 89)


if __name__ == "__main__":
    run_experiment(cuda=False)
