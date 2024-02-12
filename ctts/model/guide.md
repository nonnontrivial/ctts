# sky brightness model dataframe guide

## 1. write `./data/gan_mn.csv`

- [pull open meteo image, download & mount volumes, run container]( https://github.com/open-meteo/open-data/tree/main/tutorial_download_era5)
- `python -m ctts.model.build_dataframe`
- verify `./data/gan_mn.csv` exists

## 2. train model

- `python -m ctts.model.train_on_dataframe`
- verify `./data/model.pth` exists
