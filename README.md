# train_lm

This is some sample code to train an RNN or Transformer language model, based on the [official example from PyTorch](https://github.com/pytorch/examples/tree/master/word_language_model). I made a few changes to that code.

* I use [`torch.optim.SGD`](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) to optimize the network, whereas the original code implements SGD manually.
* I use [torchtext](https://pytorch.org/text/) to process the data. This allows the code to be automatically compatible with the PTB, Wikitext-2, and Wikitext-103 datasets.
* I include a script for Bayesian hyperparameter tuning using [Ax](https://ax.dev/).

The usage is the same as in the original code.