
import torch
import random
import numpy as np


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def clean_data(data: dict):
    for elem in data.items():
        del elem
