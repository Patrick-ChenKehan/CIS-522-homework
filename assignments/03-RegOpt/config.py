from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, RandomRotation


class CONFIG:
    batch_size = 64
    num_epochs = 10
    initial_learning_rate = 0.001
    initial_weight_decay = 0.01

    lrs_kwargs = {"T": 1000, "eta_min": 1e6}

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            # RandomRotation(15),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
