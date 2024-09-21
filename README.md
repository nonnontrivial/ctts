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
(sky brightness) available in a [public dataset](http://www.unihedron.com/projects/darksky/database/?csv=true) using
pytorch.

Another component uses [H3](https://uber.github.io/h3-py/intro.html)
to discretize the surface of the earth into cells, which form the basis
of the requests made to the api server serving up the sky brightness.
This component then pushes brightness values from the api server onto
rabbitmq, while another component consumes them from the queue -
inserting to postgres to enable historical lookup of brightness values.

## how to run

This will spin up the process of the prediction producer container
repeatedly asking the api container for sky brightness measurements
_at the current time_ across all [resolution 0 h3 cells](https://h3geo.org/docs/core-library/restable/);
publishing them to rabbitmq, which the consumer container reads from
and stores messages (in postgres container).

```shell
# create the volume for weather data
docker volume create --name open-meteo-data

# get latest data into the above volume
./update-open-meteo.sh

# run the containers
docker-compose up --build
```

Rabbitmq will take time to start up, at which time `producer` and
`consumer` containers will attempt restart to form connection.
Once rabbitmq does start, there should be output like this:

```shell
producer-1   | 2024-07-27 23:48:09,301 [INFO] publishing brightness message {'uuid': '86967a4d-d6a5-4421-8db1-73d9d0d45ea5', 'lat': 11.509775527199592, 'lon': -55.499062349013, 'h3_id': '805ffffffffffff', 'utc_iso': '2024-07-27T23:48:09.301613', 'utc_ns': 1722124089301613056, 'mpsas': 8.6264, 'model_version': '0.1.0'}
producer-1   | 2024-07-27 23:48:09,304 [INFO] 805ffffffffffff has had 2 predictions published
consumer-1   | 2024-07-27 23:48:09,321 [INFO] inserting brightness message for 805ffffffffffff
consumer-1   | 2024-07-27 23:48:09,327 [INFO] broadcasting to 0 websocket clients on consumer:8090
api-1        | 2024-07-27 23:48:09,922 [INFO] 172.19.0.7:43300 - "GET /api/v1/predict?lat=16.702868303031234&lon=-13.374845104752373 HTTP/1.1" 200
producer-1   | 2024-07-27 23:48:09,926 [INFO] HTTP Request: GET http://api:8000/api/v1/predict?lat=16.702868303031234&lon=-13.374845104752373 "HTTP/1.1 200 OK"
producer-1   | 2024-07-27 23:48:09,927 [INFO] publishing brightness message {'uuid': '01d5912d-752b-4611-8de9-4f092dc21c5c', 'lat': 16.702868303031234, 'lon': -13.374845104752373, 'h3_id': '8055fffffffffff', 'utc_iso': '2024-07-27T23:48:09.927243', 'utc_ns': 1722124089927243008, 'mpsas': 4.5332, 'model_version': '0.1.0'}
producer-1   | 2024-07-27 23:48:09,930 [INFO] 8055fffffffffff has had 3 predictions published
consumer-1   | 2024-07-27 23:48:09,945 [INFO] inserting brightness message for 8055fffffffffff
consumer-1   | 2024-07-27 23:48:09,950 [INFO] broadcasting to 0 websocket clients on consumer:8090
api-1        | 2024-07-27 23:48:10,540 [INFO] 172.19.0.7:43300 - "GET /api/v1/predict?lat=-7.460529604384309&lon=84.45314174117765 HTTP/1.1" 200
producer-1   | 2024-07-27 23:48:10,544 [INFO] HTTP Request: GET http://api:8000/api/v1/predict?lat=-7.460529604384309&lon=84.45314174117765 "HTTP/1.1 200 OK"
producer-1   | 2024-07-27 23:48:10,547 [INFO] publishing brightness message {'uuid': 'd015ee51-d60e-4722-9fb3-f442fd3c52e3', 'lat': -7.460529604384309, 'lon': 84.45314174117765, 'h3_id': '8087fffffffffff', 'utc_iso': '2024-07-27T23:48:10.547142', 'utc_ns': 1722124090547142144, 'mpsas': 9.0234, 'model_version': '0.1.0'}
producer-1   | 2024-07-27 23:48:10,549 [INFO] 8087fffffffffff has had 4 predictions published
consumer-1   | 2024-07-27 23:48:10,565 [INFO] inserting brightness message for 8087fffffffffff
consumer-1   | 2024-07-27 23:48:10,572 [INFO] broadcasting to 0 websocket clients on consumer:8090
```
