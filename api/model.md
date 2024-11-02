# model

## building and training

The api depends on a model being trained from csv data.

1. use `python -m api.model.build` to write the csv that the model trains on
2. use `python -m api.model.train` to generate a new `model.pth` (i.e. to learn new parameters)
