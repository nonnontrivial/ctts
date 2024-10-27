# CTTS

> note: this is alpha software; apis may change quickly, and quality of the brightness prediction is still being ironed out

CTTS is an open source application for reading [sky brightness](https://en.wikipedia.org/wiki/Sky_brightness) all over the
earth, without a sensor.

## features

* gRPC api for predicting sky brightness

* gRPC api for light pollution values (in RGBA, from a 2022 map)

* publisher component that repeatedly generates & stores readings for coordinates of H3 cells

## todos

- [x] support for continents other than north america
- [x] less noisy container startup
- [ ] live updates to open meteo data while app is running
- [ ] REST apis in addition to the gRPC ones
- [x] better storage of predictions in order to faciliate grouping/sorting

## about

This project is motivated by the desire for synoptic knowledge of "where are the stars good".

It would be infeasible to have [sensors](http://unihedron.com/projects/darksky/TSL237-E32.pdf)
everywhere that a brightness measurement is desired, so it would make sense to have a way of
doing inference of this value.


The approach this project takes is to use pytorch to capture the relationships in the [Globe At Night
dataset](https://globeatnight.org/maps-data/) and use that to predict sky brightness for H3
cells at resoultion 6.

> note: currently limited to north america

## running with docker

- `git clone`
- `cd` into the repo
- run the following:

```shell
# create the volume for weather data
docker volume create --name open-meteo-data

# get latest data into the above volume
./update-open-meteo.sh

# run the containers (`build` flag only necessary for first run)
docker compose up --build
```


## documentation

- [api client usage](./api/README.md)

## licensing

This project is licensed under the AGPL-3.0 license.
