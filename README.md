# ctts

This project was inspired by wanting **a way of seeing what the
[sky brightness](https://en.wikipedia.org/wiki/Sky_brightness)
is over the entire earth surface, and how it changes over time**..

Given that it would be infeasible to have [sensors](http://unihedron.com/projects/darksky/TSL237-E32.pdf)
everywhere that we want a brightness measurement, it would make
sense to have a way of performing this measurement indirectly.

---

The approach this project takes is to model the relationship
between a set of independent variables and the dependent variable
(sky brightness) available in a [public dataset](https://globeatnight.org/maps-data/) using
pytorch.

Another component uses [H3](https://uber.github.io/h3-py/intro.html)
to discretize the surface of the earth into cells, which form the basis
of the requests made to the api server serving up the sky brightness.
This component then pushes brightness values from the api server onto
rabbitmq, while another component consumes them from that queue -
inserting records into postgres to enable historical lookup.

## running with docker

This will spin up the process of the prediction producer container
repeatedly asking the api container for sky brightness predictions
_for the current time_ across all [resolution 6 H3 cells](https://h3geo.org/docs/core-library/restable/);
publishing those predictions to rabbitmq, which the consumer container reads from,
storing those messages in the `brightnessobservation` table.

> note: at present only cells in north america are generated.

- clone the repo
- `cd` into it
- run the following:

```shell
# create the volume for weather data
docker volume create --name open-meteo-data

# get latest data into the above volume
./update-open-meteo.sh

# run the containers (`build` flag only necessary for first run)
docker compose up --build
```

Rabbitmq will take time to start up, at which time `producer` and
`consumer` containers will attempt restart to form their connections
to the queue.

Once rabbitmq does start, there should be output like this:

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

This output indicates that the producer service is successfully getting
sky brightness predictions for H3 cells and that the consumer service
is storing them in the postgres table `brightnessobservation`.


## licensing

This project is licensed under the AGPL-3.0 license.

Note: The GeoJSON file located at `./pp/pp/cells/north-america.geojson` is licensed under the Apache License, Version 2.0, and retains its original copyright (Copyright 2018 Esri).

