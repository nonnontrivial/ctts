# api

api server for [sky brightness](https://en.wikipedia.org/wiki/Sky_brightness) at valid coordinates.

## Building and training the sky brightness model

The api server depends on a model being trained from augmented csv data.

These commands will generate a new `model.pth`.

- `python -m api.model.build` to write the csv that the model trains on
- `python -m api.model.train` to train on the data in the csv

## HTTP APIs

### endpoints

#### `/api/v1/predict`

Gets the predicted sky brightness at `lat` and `lon` for the current time.

```sh
curl "http://localhost:8000/api/v1/predict?lat=-30.2466&lon=-70.7494"

```

```json
{
  "mpsas": 22.0388
}
```

#### `/api/v1/pollution`

Gets the approximate artificial sky brightness
map [RGBA pixel value](https://djlorenz.github.io/astronomy/lp2022/colors.html) for a lat and lon (for the year 2022).

```sh
curl "localhost:8000/api/v1/pollution?lat=40.7277478&lon=-74.0000374"
```

```json
{
  "r": 255,
  "g": 255,
  "b": 255,
  "a": 255
}
```

### swagger ui

Open the [ui](http://localhost:8000/docs) in a browser.

