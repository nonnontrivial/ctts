import pdb
from typing import Any
from pathlib import Path

import numpy as np
from tinygrad import Tensor, nn
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters

from .build_dataframe import gan_mn_dir, gan_mn_dataframe_filename

class TinyNet:
    def __init__(self) -> None:
        self.l1 = nn.Linear(784,128,bias=False)
        self.l2 = nn.Linear(128,1,bias=False)
    def __call__(self, x:Tensor) -> Tensor:
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x

net = TinyNet()
opt = SGD(get_parameters(net), lr=3e-4)

X_train = Tensor.full(shape=(100, 6),fill_value=2)
y_train = Tensor.arange(start=0,stop=10,step=1)

X_test = Tensor.full(shape=(100, 6),fill_value=2)
y_test = Tensor.arange(start=0,stop=10,step=1)

def main() -> None:
    path_to_dataframe_file = gan_mn_dir.parent / gan_mn_dataframe_filename
    if not path_to_dataframe_file.exists():
        raise FileNotFoundError(f"!no dataframe file at {path_to_dataframe_file}")
    with Tensor.train():
        for step in range(1000):
            samp = np.random.randint(0,2,size=(64))
            pdb.set_trace()
            # batch = Tensor(X_train[samp], requires_grad=True)
            # labels = Tensor(Y_train)
            if step % 100 == 0:
                pass

if __name__ == "__main__":
    main()
