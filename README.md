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
producer-1   | 2024-10-01 00:01:38,305 [INFO] HTTP Request: GET http://api:8000/api/v1/predict?lat=50.103201482241325&lon=-143.47849001502516 "HTTP/1.1 200 OK"
producer-1   | 2024-10-01 00:01:38,306 [INFO] publishing brightness observation for cell 801dfffffffffff
api-1        | 2024-10-01 00:01:38,304 [INFO] 172.18.0.9:39216 - "GET /api/v1/predict?lat=50.103201482241325&lon=-143.47849001502516 HTTP/1.1" 200
consumer-1   | 2024-10-01 00:01:38,309 [INFO] saving brightness observation 801dfffffffffff:62f41905-fc7a-4003-a1a8-5bf8b2d739ae
producer-1   | 2024-10-01 00:01:38,314 [INFO] cell 801dfffffffffff has had 6 predictions published
producer-1   | 2024-10-01 00:01:38,970 [INFO] HTTP Request: GET http://api:8000/api/v1/predict?lat=24.053793264068165&lon=130.21990279877684 "HTTP/1.1 200 OK"
producer-1   | 2024-10-01 00:01:38,971 [INFO] publishing brightness observation for cell 804bfffffffffff
api-1        | 2024-10-01 00:01:38,967 [INFO] 172.18.0.9:39216 - "GET /api/v1/predict?lat=24.053793264068165&lon=130.21990279877684 HTTP/1.1" 200
producer-1   | 2024-10-01 00:01:38,974 [INFO] cell 804bfffffffffff has had 6 predictions published
consumer-1   | 2024-10-01 00:01:38,978 [INFO] saving brightness observation 804bfffffffffff:efbc5442-60b3-403e-b698-e8cf3cbd450a
producer-1   | 2024-10-01 00:01:39,642 [INFO] HTTP Request: GET http://api:8000/api/v1/predict?lat=-0.7617301194234768&lon=-21.43378831072749 "HTTP/1.1 200 OK"
producer-1   | 2024-10-01 00:01:39,643 [INFO] publishing brightness observation for cell 807dfffffffffff
producer-1   | 2024-10-01 00:01:39,645 [INFO] cell 807dfffffffffff has had 6 predictions published
api-1        | 2024-10-01 00:01:39,640 [INFO] 172.18.0.9:39216 - "GET /api/v1/predict?lat=-0.7617301194234768&lon=-21.43378831072749 HTTP/1.1" 200
consumer-1   | 2024-10-01 00:01:39,651 [INFO] saving brightness observation 807dfffffffffff:d6776046-b315-42bb-b472-7383b29e723e
producer-1   | 2024-10-01 00:01:40,328 [INFO] HTTP Request: GET http://api:8000/api/v1/predict?lat=-39.547652536884&lon=-36.364248231710086 "HTTP/1.1 200 OK"
producer-1   | 2024-10-01 00:01:40,329 [INFO] publishing brightness observation for cell 80c5fffffffffff
api-1        | 2024-10-01 00:01:40,327 [INFO] 172.18.0.9:39216 - "GET /api/v1/predict?lat=-39.547652536884&lon=-36.364248231710086 HTTP/1.1" 200
producer-1   | 2024-10-01 00:01:40,331 [INFO] cell 80c5fffffffffff has had 6 predictions published
consumer-1   | 2024-10-01 00:01:40,334 [INFO] saving brightness observation 80c5fffffffffff:d4713ee8-d370-4f71-9572-43dc81844e6c
```
