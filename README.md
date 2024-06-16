# ctts

closer to the stars. application for predictive sky brightness over discretized earth.

## getting started

this will spin up the process of the prediction producer container repeatedly asking the api server for sky brightness
measurements across all [resolution 0 h3 cells](https://h3geo.org/docs/core-library/restable/).

```shell
docker-compose up --build
```
