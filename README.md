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

# run the containers (optionally use `build` flag)
docker compose up --build
```

After rabbitmq starts up, the producer and consumer containers will start up,
at which point you should see output like this:

```sh
producer-1   | 2024-10-28 01:44:02,803 [INFO] publishing {'uuid': 'ef566c44-78db-45e3-8558-7c35ed1e5095', 'lat': 39.6421440381801, 'lon': 44.21375098521473, 'h3_id': '802dfffffffffff', 'mpsas': 12.076292037963867, 'timestamp_utc': '2024-10-28T01:44:02.802762+00:00'} to brightness.prediction
producer-1   | 2024-10-28 01:44:02,804 [INFO] publishing {'start_time_utc': '2024-10-28T01:43:58.223292+00:00', 'end_time_utc': '2024-10-28T01:44:02.804525+00:00', 'duration_s': 4} to brightness.cycle
consumer-1   | 2024-10-28 01:44:02,810 [INFO] {'uuid': '91b939ae-fe15-4e0c-abcb-3104a5ef0644', 'lat': 28.50830365117256, 'lon': 86.00509004642792, 'h3_id': '803dfffffffffff', 'mpsas': 20.70200538635254, 'timestamp_utc': datetime.datetime(2024, 10, 28, 1, 44, 2, 271912, tzinfo=datetime.timezone.utc)}
```


## documentation

- [api client usage](./api/README.md)

## licensing

This project is licensed under the AGPL-3.0 license.
