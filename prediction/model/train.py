"""Train the sky brightness model.

python -m prediction.model.train
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ..constants import features
from ..nn import NeuralNetwork

GAN_FILENAME = "globe_at_night.csv"

HIDDEN_SIZE = 64 * 3
OUTPUT_SIZE = 1
FEATURES_SIZE = len(features)

print("loading dataframe..")
cwd = Path.cwd()
df = pd.read_csv(cwd / "data" / GAN_FILENAME)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_printoptions(sci_mode=False)

feature_tensor = torch.tensor(df[features].values.astype(np.float32))
feature_tensor = torch.nan_to_num(feature_tensor, nan=0.0)
target_tensor = torch.tensor(df["SQMReading"].values.astype(np.float32)).to(
    torch.float32
)

data_tensor = TensorDataset(feature_tensor, target_tensor)
train_size = int(0.8 * len(data_tensor))
test_size = len(data_tensor) - train_size
train_tensor, test_tensor = random_split(data_tensor, [train_size, test_size])

train_dataloader = DataLoader(dataset=train_tensor, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=test_tensor, batch_size=16, shuffle=True)

model = NeuralNetwork().to(device)

loss_fn = nn.HuberLoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(
    data_loader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.HuberLoss,
    optimizer: torch.optim.Adam,
):
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output.squeeze(), y)
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm(model.parameters(), max_norm=5)
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            size = len(data_loader.dataset)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_model(data_loader: DataLoader, model: NeuralNetwork, loss_fn: nn.MSELoss):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch, (X, y) in enumerate(data_loader):
            pred = model(X)
            print(f"prediction at {batch} was {pred} for {X} ")
            loss = loss_fn(pred.squeeze(), y)
            test_loss += loss.item() * X.size(0)
        avg_loss = test_loss / len(data_loader.dataset)
        print(f"avg loss in test is {avg_loss}")

epochs = 100
for t in range(epochs):
    print(f"epoch {t+1}")
    train_loop(train_dataloader, model, loss_fn, optimizer)

test_model(test_dataloader, model, loss_fn)

saved_model_path = cwd / "model.pth"
print(f"saving to {saved_model_path}")
torch.save(model.state_dict(), saved_model_path)
