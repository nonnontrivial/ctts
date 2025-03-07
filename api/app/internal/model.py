from pathlib import Path
import torch
import torch.nn as nn

NUM_FEATURES = 7
HIDDEN_SIZE = 64 * 3
OUTPUT_SIZE = 1


class NN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(NUM_FEATURES, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE // 2, OUTPUT_SIZE),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


path_to_state_dict = Path(__file__).parent / "model.pth"
