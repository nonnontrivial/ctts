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
cells at a configured [H3 resoultion](https://h3geo.org/docs/core-library/restable/) (default `0`).

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
producer-1   | 2024-10-28 13:16:27,502 [INFO] publishing {'uuid': 'e6c22004-9180-4599-9e87-36b86f68a5e7', 'lat': 43.42281493904898, 'lon': -97.42465926125905, 'h3_id': '8027fffffffffff', 'mpsas': 9.942784309387207, 'timestamp_utc': '2024-10-28T13:16:27.501192+00:00'} to brightness.prediction
producer-1   | 2024-10-28 13:16:27,580 [INFO] publishing {'uuid': 'a548be90-89f7-4995-9239-197523c3afd0', 'lat': 19.093680683484372, 'lon': 43.638818828910864, 'h3_id': '8053fffffffffff', 'mpsas': 9.202325820922852, 'timestamp_utc': '2024-10-28T13:16:27.579755+00:00'} to brightness.prediction
producer-1   | 2024-10-28 13:16:27,656 [INFO] publishing {'uuid': '2759f0e8-2f94-4efd-bf19-4b765947d983', 'lat': 60.432795263055546, 'lon': -77.20705748560815, 'h3_id': '800ffffffffffff', 'mpsas': 11.305692672729492, 'timestamp_utc': '2024-10-28T13:16:27.655087+00:00'} to brightness.prediction
producer-1   | 2024-10-28 13:16:27,736 [INFO] publishing {'uuid': 'd53872da-2505-41a9-84f1-d9336b0aff83', 'lat': -30.01574044171678, 'lon': 129.95847216046155, 'h3_id': '80b9fffffffffff', 'mpsas': 12.414505004882812, 'timestamp_utc': '2024-10-28T13:16:27.735392+00:00'} to brightness.prediction
producer-1   | 2024-10-28 13:16:27,737 [INFO] publishing {'start_time_utc': '2024-10-28T13:16:24.874541+00:00', 'end_time_utc': '2024-10-28T13:16:27.737791+00:00', 'duration_s': 2} to brightness.cycle
consumer-1   | 2024-10-28 13:16:27,744 [INFO] {'uuid': 'aa89439d-9a22-41e1-b8d2-674bea5263ee', 'lat': -74.92843438917433, 'lon': -34.64375807722018, 'h3_id': '80effffffffffff', 'mpsas': 23.74591636657715, 'timestamp_utc': datetime.datetime(2024, 10, 28, 13, 16, 25, 624186, tzinfo=datetime.timezone.utc)}
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
