"""
Trains a language model from the command line.
"""
import argparse
import random

import torch

from data import load_data
from train import run_experiment

# Command line arguments
parser = argparse.ArgumentParser(description="RNN/Transformer LM "
                                             "Hyperparameters")

parser.add_argument("--data", type=str, default="WikiText2",
                    help="Dataset: WikiText2 or WikiText103")
parser.add_argument("--model", type=str, default="LSTM",
                    help="Network architecture: RNN_TANH, RNN_RELU, LSTM, "
                         "GRU, or Transformer")
parser.add_argument("--emsize", type=int, default=200,
                    help="Embedding size")
parser.add_argument("--nhid", type=int, default=200,
                    help="Hidden size")
parser.add_argument("--nlayers", type=int, default=2,
                    help="Number of RNN or Transformer layers")
parser.add_argument("--lr", type=float, default=20,
                    help="Initial learning rate for SGD")
parser.add_argument("--clip", type=float, default=0.25,
                    help="Amount of gradient clipping")
parser.add_argument("--epochs", type=int, default=40,
                    help="Maximum number of epochs")
parser.add_argument("--batch_size", type=int, default=20, metavar="N",
                    help="Batch size")
parser.add_argument("--bptt", type=int, default=35,
                    help="Maximum sequence length for BPTT")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout (0 = no dropout)")
parser.add_argument("--tied", action="store_true",
                    help="Set this flag to tie the word embedding and softmax "
                         "weights")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed")
parser.add_argument("--cuda", action="store_true",
                    help="Set this flag to use CUDA")
parser.add_argument("--log-interval", type=int, default=200, metavar="N",
                    help="The frequency of reporting")
parser.add_argument("--save", type=str, default="model.pt",
                    help="Path to save the final model to")
parser.add_argument("--nhead", type=int, default=2,
                    help="The number of attention heads in a Transformer "
                         "model")

kwargs = vars(parser.parse_args())

# Set seed
if kwargs["seed"] is not None:
    random.seed(kwargs["seed"])
del kwargs["seed"]

# Load data
device = torch.device("cuda" if kwargs["cuda"] else "cpu")
train_iter, val_iter, test_iter = load_data(kwargs["data"], device,
                                            kwargs["batch_size"],
                                            kwargs["bptt"])
del kwargs["data"]
del kwargs["cuda"]

run_experiment(device=device, train_iter=train_iter, val_iter=val_iter,
               test_iter=test_iter, **kwargs)
