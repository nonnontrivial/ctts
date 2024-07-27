# ctts

This project was inspired by wanting **a way of seeing what the
[sky brightness](https://en.wikipedia.org/wiki/Sky_brightness) is at different points around the globe, and how that is
changing over time**..

Needless to say, we cannot have sensors for this sort of thing
everwhere we would want a measurement, therefore we need a way
of performing this measurement
in a less direct way..

---

The approach this project takes is to model the relationship
between a set of independent variables and the dependent variable
(sky brightness) available in a [public dataset](http://www.unihedron.com/projects/darksky/database/?csv=true).

H3 is then used to discretize cells over the earth that form
the basis of the requests made to the api server serving up
the sky brightness.

Another component pushes brightness values from the model onto
rabbitmq while saving them to postgres, to enable historical lookup.

## running locally

this will spin up the process of the prediction producer container repeatedly asking the api server for sky brightness
measurements across all [resolution 0 h3 cells](https://h3geo.org/docs/core-library/restable/) and publishing to
rabbitmq, which the consumer container reads from and stores in postgres.

```shell
# create the volume for weather data
docker volume create --name open-meteo-data

# get latest data into the above volume
./update-open-meteo.sh

# run the containers
docker-compose up --build
```
