# CTTS

> note: this is alpha software; apis may change quickly, and quality of the brightness prediction is still being ironed out

CTTS is an open source application for reading [sky brightness](https://en.wikipedia.org/wiki/Sky_brightness) all over the
earth, without a sensor.

## features

* gRPC api for predicting sky brightness

* gRPC api for light pollution values (in RGBA, from a 2022 map)

* publisher component that repeatedly generates readings for coordinates of H3 cells

* consumer component that stores the readings and computes the reading with highest `mpsas` during the last cycle of observation

## todos

- [x] support for continents other than north america
- [x] less noisy container startup
- [ ] live updates to open meteo data while app is running
- [ ] REST apis in addition to the gRPC ones
- [x] better storage of predictions in order to faciliate grouping/sorting

## about

This project is motivated by the desire for synoptic knowledge of "where are the stars good".

It would be infeasible to have [sensors](http://unihedron.com/projects/darksky/TSL237-E32.pdf)
everywhere you would want a brightness measurement, so it would make sense to have a way of
doing inference of this value.


The approach this project takes is to use pytorch to capture the relationships in the [Globe At Night
dataset](https://globeatnight.org/maps-data/) and use that to predict sky brightness for H3
cells at a configured [H3 resolution](https://h3geo.org/docs/core-library/restable/) (default `0`).

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

```log
producer-1   | 2024-11-13 03:01:02,478 [INFO] publishing {'uuid': 'c6df89c5-a4fa-48fc-bfd8-11d08494902f', 'lat': 16.702868303031234, 'lon': -13.374845104752373, 'h3_id': '8055fffffffffff', 'mpsas': 6.862955570220947, 'timestamp_utc': '2024-11-13T03:01:02.478000+00:00'} to brightness.prediction
producer-1   | 2024-11-13 03:01:02,553 [INFO] publishing {'uuid': '9b5f2e8b-c22d-4d05-900e-0156f78632ce', 'lat': 26.283628653081813, 'lon': 62.954274989658984, 'h3_id': '8043fffffffffff', 'mpsas': 9.472949028015137, 'timestamp_utc': '2024-11-13T03:01:02.552848+00:00'} to brightness.prediction
producer-1   | 2024-11-13 03:01:02,625 [INFO] publishing {'uuid': 'fbbc3cd5-839d-43de-a7c4-8f51100679fd', 'lat': -4.530154895350926, 'lon': -42.02241568705745, 'h3_id': '8081fffffffffff', 'mpsas': 9.065463066101074, 'timestamp_utc': '2024-11-13T03:01:02.624759+00:00'} to brightness.prediction
producer-1   | 2024-11-13 03:01:02,626 [INFO] publishing {'start_time_utc': '2024-11-13T03:01:00.114586+00:00', 'end_time_utc': '2024-11-13T03:01:02.626208+00:00', 'duration_s': 2} to brightness.cycle
consumer-1   | 2024-11-13 03:01:02,631 [INFO] cycle completed with {'uuid': '4bb0c627-596c-42be-a93a-26f36c5ca3c1', 'lat': 55.25746462939812, 'lon': 127.08774514928741, 'h3_id': '8015fffffffffff', 'mpsas': 23.763256072998047, 'timestamp_utc': datetime.datetime(2024, 11, 13, 3, 1, 1, 129155, tzinfo=datetime.timezone.utc)}
```

The above output means:

1. the producer container is publishing the brightness readings it is getting from
the api container

2. the consumer container has determined which reading made during the last cycle
through H3 cells had the highest brightness (`mpsas` is the measure of brightness
spread over a square arcsecond of sky, where higher means darker sky with more
stars visible)

## changing the resolution

By default, the producer runs over all resolution 0 cells, but this can be adjusted
by setting the environment of the `producer` container in `docker-compose.yaml`:

```yaml
producer:
  build: ./pp
  environment:
    RESOLUTION: 0
```


## documentation

- [how to write your own client for the sky brightness gRPC api](./api/README.md)

## licensing

This project is licensed under the AGPL-3.0 license.
