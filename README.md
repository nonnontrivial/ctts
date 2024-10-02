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
inserting records in postgres to enable historical lookup.

## running with docker

This will spin up the process of the prediction producer container
repeatedly asking the api container for sky brightness predictions
_for the current time_ across all [resolution 6 H3 cells](https://h3geo.org/docs/core-library/restable/);
publishing those predictions to rabbitmq, which the consumer container reads from,
storing those messages in the `brightnessobservation` table.

> note: at present only cells in north america are generated.

```shell
# create the volume for weather data
docker volume create --name open-meteo-data

# get latest data into the above volume
./update-open-meteo.sh

# run the containers (build flag only necessary for first run)
docker compose up --build
```

Rabbitmq will take time to start up, at which time `producer` and
`consumer` containers will attempt restart to form their connections
to the queue.

Once rabbitmq does start, there should be output like this:

```log
api-1        | 2024-10-01 23:13:35,963 [INFO] 172.18.0.8:52564 - "GET /api/v1/predict?lat=18.93443063422478&lon=-102.69523192458102 HTTP/1.1" 200
producer-1   | 2024-10-01 23:13:35,965 [INFO] publishing brightness observation for cell 8049fffffffffff
producer-1   | 2024-10-01 23:13:35,970 [INFO] cell 8049fffffffffff has had 61 predictions published
consumer-1   | 2024-10-01 23:13:35,975 [INFO] saved brightness observation 8049fffffffffff:8d3adf35-6d65-4033-b901-5500994440f7
producer-1   | 2024-10-01 23:13:36,072 [INFO] 84 distinct cells have observations published
api-1        | 2024-10-01 23:13:36,229 [INFO] 172.18.0.8:52564 - "GET /api/v1/predict?lat=23.751984470800828&lon=-98.56249212703489 HTTP/1.1" 200
producer-1   | 2024-10-01 23:13:36,234 [INFO] HTTP Request: GET http://api:8000/api/v1/predict?lat=23.751984470800828&lon=-98.56249212703489 "HTTP/1.1 200 OK"
producer-1   | 2024-10-01 23:13:36,235 [INFO] publishing brightness observation for cell 8049fffffffffff
producer-1   | 2024-10-01 23:13:36,236 [INFO] cell 8049fffffffffff has had 62 predictions published
consumer-1   | 2024-10-01 23:13:36,244 [INFO] saved brightness observation 8049fffffffffff:7a0de422-00c6-42da-b771-0ad740d72472
producer-1   | 2024-10-01 23:13:36,338 [INFO] 85 distinct cells have observations published
api-1        | 2024-10-01 23:13:36,494 [INFO] 172.18.0.8:52564 - "GET /api/v1/predict?lat=17.630481352179363&lon=-100.25972697954386 HTTP/1.1" 200
producer-1   | 2024-10-01 23:13:36,496 [INFO] HTTP Request: GET http://api:8000/api/v1/predict?lat=17.630481352179363&lon=-100.25972697954386 "HTTP/1.1 200 OK"
producer-1   | 2024-10-01 23:13:36,498 [INFO] publishing brightness observation for cell 8049fffffffffff
producer-1   | 2024-10-01 23:13:36,513 [INFO] cell 8049fffffffffff has had 63 predictions published
consumer-1   | 2024-10-01 23:13:36,522 [INFO] saved brightness observation 8049fffffffffff:35b65bbe-67dc-4547-a1dd-9a6eea059828
producer-1   | 2024-10-01 23:13:36,616 [INFO] 86 distinct cells have observations published
api-1        | 2024-10-01 23:13:36,787 [INFO] 172.18.0.8:52564 - "GET /api/v1/predict?lat=27.38982117748413&lon=-101.71565690812395 HTTP/1.1" 200
producer-1   | 2024-10-01 23:13:36,795 [INFO] HTTP Request: GET http://api:8000/api/v1/predict?lat=27.38982117748413&lon=-101.71565690812395 "HTTP/1.1 200 OK"
producer-1   | 2024-10-01 23:13:36,796 [INFO] publishing brightness observation for cell 8049fffffffffff
producer-1   | 2024-10-01 23:13:36,801 [INFO] cell 8049fffffffffff has had 64 predictions published
consumer-1   | 2024-10-01 23:13:36,808 [INFO] saved brightness observation 8049fffffffffff:0a346078-002b-4bce-86d5-b38455e9fbac
producer-1   | 2024-10-01 23:13:36,902 [INFO] 87 distinct cells have observations published
```

This output indicates that the producer service is sucessfully
fetching sky brightness predictions for H3 cells and the consumer
service is storing them in the postgres table `brightnessobservation`.


## licensing

This project is licensed under the AGPL-3.0 license.

Note: The GeoJSON file located at `./pp/pp/cells/north-america.geojson` is licensed under the Apache License, Version 2.0, and retains its original copyright (Copyright 2018 Esri).

