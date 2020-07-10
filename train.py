import math
import time
from argparse import Namespace
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

    :param h:
    :return:
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


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
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        if isinstance(model, TransformerModel):
            output = model(batch.text)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(batch.text, hidden)

        loss = criterion(output, batch.target.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print("| epoch {:3d} | {:5d}/{:5d} batches | "
                  "ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}"
                  "".format(epoch, i, len(train_iter), elapsed * 1000 /
                            log_interval, cur_loss, 0))
            total_loss = 0
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
        hidden = model.init_hidden(10)

    with torch.no_grad():
        for batch in eval_iter:
            if isinstance(model, TransformerModel):
                output = model(batch.text)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(batch.text, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(batch) * criterion(output, batch.target).item()
    return total_loss / (len(eval_iter) - 1)


def run_experiment(args: Namespace):
    """
    Trains a language model.

    :param args: Parsed command-line arguments
    :return: None
    """
    device = torch.device("cuda" if args.cuda else "cpu")

    # Load data
    text_field = tt.Field()
    train_data, valid_data, test_data = WikiText2.splits(text_field)
    text_field.build_vocab(train_data)

    train_iter = tt.BPTTIterator(train_data, args.batch_size, args.bptt,
                                 device=0, repeat=False)
    val_iter = tt.BPTTIterator(valid_data, 10, args.bptt, device=0,
                               repeat=False)
    test_iter = tt.BPTTIterator(test_data, 10, args.bptt, device=0,
                                repeat=False)

    # Build the model
    ntokens = len(text_field.vocab)
    if args.model == "Transformer":
        model = TransformerModel(ntokens, args.emsize, args.nhead,
                                 args.nhid,
                                 args.nlayers, args.dropout).to(device)
    else:
        model = RNNModel(args.model, ntokens, args.emsize, args.nhid,
                         args.nlayers, args.dropout, args.tied).to(
            device)

    criterion = nn.CrossEntropyLoss()

    # Epoch loop
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Train and validate
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train(model, train_iter, epoch, criterion, optimizer, args.clip,
              1) # args.log_interval)
        val_loss = evaluate(model, val_iter, criterion)

        # Report validation loss
        epoch_time = time.time() - epoch_start_time
        print("-" * 89)
        print("| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
              "valid ppl {:8.2f}".format(epoch, epoch_time, val_loss,
                                         math.exp(val_loss)))
        print("-" * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, "wb") as f:
                torch.save(model, f)
            best_val_loss = val_loss

    # Load the best model
    with open(args.save, "rb") as f:
        model = torch.load(f)
    if isinstance(model, RNNModel):
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(model, test_iter, criterion)
    print("=" * 89)
    print("| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
        test_loss, math.exp(test_loss)))
    print("=" * 89)
