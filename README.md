# CTTS

> note: this is _alpha software_; apis may change quickly, and quality of the brightness prediction is still being ironed out

CTTS is an open source application for reading [sky brightness](https://en.wikipedia.org/wiki/Sky_brightness) all over the earth's landmass, without a sensor.

## about

This project is motivated by wanting an answer to the following question:

> how much better are the stars in the [Atacama](https://en.wikipedia.org/wiki/Atacama_Desert) than in my backyard?

It would be infeasible to have [sensors](http://unihedron.com/projects/darksky/TSL237-E32.pdf)
everywhere you would want a sky brightness measurement, but one path is to do inference of this value.

The approach this project takes is:

- use pytorch to capture the relationship between various factors and sky brightness within a [public dataset](https://globeatnight.org/maps-data/)
- put that behind an gRPC API
- discretize the world using [H3](https://h3geo.org/docs/core-library/restable/)
- continuously get predicted sky brightness for those H3 cells from the gRPC API

## todos

- [ ] update open meteo data while app is running
- [ ] create typer app to manage brightness model
- [ ] brightness prediction cycle can push alerts

## running with docker

- `git clone`
- `cd` into the repo
- run the following:

```shell
# create the volume for weather data
docker volume create --name open-meteo-data

# get latest data into the above volume
./update-open-meteo-data.sh

# build and run the containers
docker compose up --build
```

After rabbitmq starts up, the producer and consumer containers will start up,
at which point you should see output like this:

```log
producer-1   | 2024-12-21 17:08:55,237 [INFO] publishing {'uuid': '0cdacdcb-dcf3-4d5c-9e60-94d397d89840', 'lat': 69.66345294982115, 'lon': -30.968044606549025, 'h3_id': '8007fffffffffff', 'mpsas': 24.703824996948242, 'timestamp_utc': '2024-12-21T17:08:55.236185+00:00'} to brightness.prediction
producer-1   | 2024-12-21 17:08:55,355 [INFO] publishing {'uuid': 'f16a7b7c-039d-44d6-b764-fc37fadad1b7', 'lat': 26.80710329336693, 'lon': 109.167486033384, 'h3_id': '8041fffffffffff', 'mpsas': 10.82265853881836, 'timestamp_utc': '2024-12-21T17:08:55.354661+00:00'} to brightness.prediction
producer-1   | 2024-12-21 17:08:55,356 [INFO] publishing {'start_time_utc': '2024-12-21T17:08:34.174937+00:00', 'end_time_utc': '2024-12-21T17:08:55.356353+00:00', 'duration_s': 21} to brightness.cycle
producer-1   | 2024-12-21 17:08:55,502 [INFO] publishing {'uuid': 'bc236db7-dd78-43cb-925b-78ea7c777f5e', 'lat': 16.702868303031234, 'lon': -13.374845104752373, 'h3_id': '8055fffffffffff', 'mpsas': 6.5024333000183105, 'timestamp_utc': '2024-12-21T17:08:55.501490+00:00'} to brightness.prediction
consumer-1   | 2024-12-21 17:08:55,507 [INFO] cycle completed with max observation {'uuid': '0fbfe7cd-4b49-49b3-9c51-b5560706a2d8', 'lat': -69.66345294982115, 'lon': 149.03195539345094, 'h3_id': '80edfffffffffff', 'mpsas': 28.068134307861328, 'timestamp_utc': datetime.datetime(2024, 12, 21, 17, 8, 53, 2272, tzinfo=datetime.timezone.utc)}
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
