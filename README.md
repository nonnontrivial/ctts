# CTTS

> n.b.: this is _alpha software_; apis may change quickly, and quality of the brightness prediction is still being ironed out

CTTS is an open source application for reading [sky brightness](https://en.wikipedia.org/wiki/Sky_brightness) without a sensor.

It works by using a model trained on [GaN](https://globeatnight.org/maps-data/) data to do inference of sky brightness in terms of [H3 cells](https://h3geo.org).

## features

- api server for sky brightness at given H3 cells
- continuous "snapshots" of sky brightness over H3 cells in given geojson file

## run

To continuously generate snapshots of sky brightness over H3 cells in given geojson file:

1. clone the repo
2. sync openmeteo data: `./sync-open-meteo-data.sh`
3. create `./snapshot/data.geojson` (cells will be made from the exterior space of polygons in this file)
4. run the containers: `docker compose up`

> e.g. logs in docker should look like this:

```log
api-1        |       INFO   172.18.0.5:53506 - "POST /infer HTTP/1.1" 200
snapshot-1   | 2025-03-08 15:45:26,085 - INFO - HTTP Request: POST http://api/infer "HTTP/1.1 200 OK"
snapshot-1   | 2025-03-08 15:45:26,089 - INFO - published data for 30 cells to brightness.snapshot
```

5. hook into this data by running one of the consumer scripts in `./consumers/`

### consumers

> n.b. examples are provided in `./consumers/`

**consumers** are just a way of hooking into brightness data published by the snapshot container.

The messages coming over the `brightness.snapshot` queue are JSON objects with the following structure:

```json
{
  "generated_in": 50.59500000399453,
  "completed_at": "2025-03-20 12:37:51.516",
  "units": {
    "inferred_brightnesses": "mpsas",
    "generated_in": "ms"
  },
  "inferred_brightnesses": {
    "8928308280fffff": 18.303659439086914
  },
  "is_night": true
}
```

### configuration

#### resolution

To adjust the H3 resolution that is used to fill the geojson geometry, edit the `RESOLUTION` env var in
the snapshot container in `./docker-compose.yml` file.

## images

the `api` image is available from ghcr.io:

```sh
docker pull ghcr.io/nonnontrivial/ctts-api:latest

```

## licensing

This project is licensed under the AGPL-3.0 license.
