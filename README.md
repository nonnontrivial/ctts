# ctts

closer to the stars. application for predictive sky brightness over discretized earth.

## getting started

this will spin up the process of the prediction producer container repeatedly asking the api server for sky brightness
measurements across all [resolution 0 h3 cells](https://h3geo.org/docs/core-library/restable/) and publishing to rabbitmq.

```shell
# create the volume for weather data
docker volume create --name open-meteo-data

# get latest data into the above volume
./update-open-meteo.sh

# run the containers
docker-compose up --build
```
