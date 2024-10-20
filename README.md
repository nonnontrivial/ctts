# CTTS

> note: this is alpha software; apis may change quickly, and quality of the brightness prediction is still being ironed out

CTTS is an open source application for reading [sky brightness](https://en.wikipedia.org/wiki/Sky_brightness) all over the
earth, without a sensor.

## features

* gRPC api for sky brightness "readings" (at the current time across H3 cells at resolution 6 in north america)

* gRPC api for light pollution values (in RGBA, from a 2022 map)

* publisher component that repeatedly generates & stores readings for coordinates of H3 cells

## todos

- [ ] support for continents other than north america
- [ ] less noisy container startup
- [ ] live updates to open meteo data while app is running
- [ ] REST apis in addition to the gRPC ones
- [ ] better storage of predictions in order to faciliate grouping/sorting

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

> Rabbitmq will take time to start up, at which time `producer` and `consumer` containers will attempt restart until the channel becomes available.
> Once rabbitmq does start, there should be output like this:

```log
producer-1   | 2024-10-03 23:59:12,266 [INFO] brightness observation response for 8649a36a7ffffff is uuid: "f275a795-8af7-491b-9645-3ce2e14fe3cd"
producer-1   | lat: 18.575429951519293
producer-1   | lon: -101.86020792493713
producer-1   | utc_iso: "2024-10-03T23:59:12.266163+00:00"
producer-1   | mpsas: 12.7519617
consumer-1   | 2024-10-03 23:59:12,269 [INFO] received message b'{"uuid": "f275a795-8af7-491b-9645-3ce2e14fe3cd", "lat": 18.575429951519293, "lon": -101.86020792493713, "h3_id": "8649a36a7ffffff", "utc_iso": "2024-10-03T23:59:12.266163+00:00", "mpsas": 12.751961708068848}'
producer-1   | 
producer-1   | 2024-10-03 23:59:12,267 [INFO] 260 distinct cells have had observations published
consumer-1   | 2024-10-03 23:59:12,276 [INFO] saved BrightnessObservation(#8649a36a7ffffff,12.751961708068848,2024-10-03T23:59:12.266163+00:00)
```

This output indicates that the producer service is successfully getting sky brightness predictions
for H3 cells and that the consumer service is storing them in the postgres table `brightnessobservation`.

`mpsas` in the response stands for 'magnitudes per square arcsecond', and it is the predicted brightness
value for that location.

## documentation

- [api client usage](./api/README.md)

## licensing

This project is licensed under the AGPL-3.0 license.

Note: The GeoJSON file located at `./pp/pp/cells/north-america.geojson` is licensed under the Apache License, Version 2.0, and retains its original copyright (Copyright 2018 Esri).

