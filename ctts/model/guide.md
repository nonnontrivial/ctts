# sky brightness model dataframe guide

predictive sky brightess happens through a model trained on augmented publicly
available data.

the data is stored in a tarball in this repo (`gan_mn.tar.gz`).

there are _3 main commands_ used to get the model in a state where it can
deliver predictions. they should usually be run in order.

they are:

- `python -m data.unpack` for unpacking the tarball mentioned above
- `python -m ctts.model.build` for augmenting the csv data from unpacked tarball
- `python -m ctts.model.train` for training the model on the csv from build step

> note: `config.ini` can be adjusted to tweak number of epochs in training
> as well as the training .csv output filename (default values are used in this guide)

## 0. load training data

- unpack training data from tarball with `python -m data.unpack`

## 1. build the training data (default `./data/gan_mn.csv`)

- [pull open meteo image, download & mount volumes, run container](https://github.com/open-meteo/open-data/tree/main/tutorial_download_era5)

```sh
# run the open meteo container, necessary for the HTTP requests
# `build_dataframe` makes for weather data
docker run -d --rm -v open-meteo-data:/app/data -p 8080:8080 ghcr.io/open-meteo/open-meteo
```

- build the dataframe with `python -m ctts.model.build`
- verify `./data/gan_mn.csv` exists

## 2. train model on training data

- `python -m ctts.model.train`
- verify `./data/model.pth` exists
