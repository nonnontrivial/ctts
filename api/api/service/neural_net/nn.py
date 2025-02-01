import torch.nn as nn

features = [
    "Latitude",
    "Longitude",
    "Elevation",
    "CloudCover",
    "UTTimeHour",
    "MoonAlt",
    "MoonAz",
]

HIDDEN_SIZE = 64 * 3
OUTPUT_SIZE = 1


class NN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(features), HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE // 2, OUTPUT_SIZE),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
