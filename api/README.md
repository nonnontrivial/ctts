# api

api server for sky brightness.

## Building and training the sky brightness model

- `python -m api.model.build` to write the csv that the model trains on
- `python -m api.model.train` to train on the data in the csv

## HTTP APIs

### how to run locally

> Note: tested on python 3.11

```sh
pip install -r requirements.txt
python -m api.main
```

### endpoints

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

### swagger ui

Open the [ui](http://localhost:8000/docs) in a browser.

