from typing import Tuple

import torch
from torchtext import data as tt
from torchtext.datasets import WikiText2, WikiText103, PennTreebank


def load_data(dataset_name: str, device: torch.device, batch_size: int,
              bptt: int) -> \
        Tuple[tt.Iterator, ...]:
    """
    Loads a dataset based on its name.

    :param dataset_name: WikiText2, WikiText103, or PennTreebank
    :param device: The device to put tensors on
    :param batch_size: The batch size
    :param bptt: The sequence length
    :return: Training, validation, and testing Iterators
    """
    text_field = tt.Field()
    if dataset_name == "WikiText2":
        train_data, valid_data, test_data = WikiText2.splits(text_field)
    elif dataset_name == "WikiText103":
        train_data, valid_data, test_data = WikiText103.splits(text_field)
    elif dataset_name == "PennTreebank":
        train_data, valid_data, test_data = PennTreebank.splits(text_field)
    else:
        raise ValueError("{} is not a valid dataset.".format(dataset_name))

    text_field.build_vocab(train_data)

    train_iter = tt.BPTTIterator(train_data, batch_size, bptt,
                                 device=device, repeat=False)
    val_iter = tt.BPTTIterator(valid_data, 10, bptt, device=device,
                               repeat=False)
    test_iter = tt.BPTTIterator(test_data, 10, bptt, device=device,
                                repeat=False)

    return train_iter, val_iter, test_iter
