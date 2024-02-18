import torch
from torch import nn

from .constants import HIDDEN_SIZE, OUTPUT_SIZE


class LinearNet(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE // 2, OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
