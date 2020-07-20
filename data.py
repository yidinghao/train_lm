from typing import Tuple

import torch
from torchtext import data as tt
from torchtext.datasets import WikiText2, WikiText103, PennTreebank


def load_data(dataset_name: str, device: torch.device) -> \
        Tuple[tt.Dataset, ...]:
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
    return train_data, valid_data, test_data
