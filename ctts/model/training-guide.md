# sky brightness model dataframe guide

> note: `config.ini` can be adjusted to tweak number of epochs in training
> as well as the training .csv output filename (default value is used in this guide)

## 1. build the training data (`./data/gan_mn.csv`)

- [pull open meteo image, download & mount volumes, run container](https://github.com/open-meteo/open-data/tree/main/tutorial_download_era5)

```sh
# run the open meteo container, necessary for the HTTP requests
# `build_dataframe` makes for weather data
docker run -d --rm -v open-meteo-data:/app/data -p 8080:8080 ghcr.io/open-meteo/open-meteo
```

- build the dataframe with `python -m ctts.model.build_dataframe`
- verify `./data/gan_mn.csv` exists

## 2. train model on training data

- `python -m ctts.model.train_on_dataframe`
- verify `./data/model.pth` exists
