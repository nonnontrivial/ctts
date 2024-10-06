# api

gRPC api for [sky brightness](https://en.wikipedia.org/wiki/Sky_brightness) at valid coordinates.

## building and training the sky brightness model

The api depends on a model being trained from csv data (`globe_at_night.csv`).

These commands will generate a new `model.pth`:

- `python -m api.model.build` to write the csv that the model trains on
- `python -m api.model.train` to train on the data in the csv
