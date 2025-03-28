# CTTS

> n.b.: this is _alpha software_; apis may change quickly, and inference quality is still being ironed out

CTTS is an open source application for reading [sky brightness](https://en.wikipedia.org/wiki/Sky_brightness) without a sensor.

It works by using a model trained on [GaN](https://globeatnight.org/maps-data/) data to do inference of sky brightness in terms of [H3 cells](https://h3geo.org).

## features

- api server for sky brightness at given H3 cells
- continuous "snapshots" of sky brightness over H3 cells in geojson

## run

To continuously generate snapshots of sky brightness over H3 cells in given geojson file:

1. clone the repo
2. sync openmeteo data: `./sync-open-meteo-data.sh`
3. run the containers: `docker compose up`
4. add geojson data using the REST endpoint:

```sh
# n.b. assumes you have already created `data.geojson`
curl -X POST -H "Content-Type: application/json" -d @data.geojson localhost:8000/geojson
```

5. logs should then begin to look like:

```log
snapshot-1   | 2025-03-22 23:37:54,313 - INFO - requesting inference for 49 cells
api-1        |       INFO   172.18.0.5:41008 - "POST /infer HTTP/1.1" 200
snapshot-1   | 2025-03-22 23:37:57,265 - INFO - HTTP Request: POST http://api/infer "HTTP/1.1 200 OK"
snapshot-1   | 2025-03-22 23:37:57,268 - INFO - published data for 49 cells to brightness.snapshot
```

6. hook into this data by running one of the consumer scripts in `./consumers/`:

```sh
uv run store_in_sqlite.py
```

## message format

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

## REST api

### `POST /infer`

```sh
curl -X POST -H "Content-Type: application/json" -d '["8928308280fffff"]' localhost:8000/infer
```

```json
{
  "generated_in": 53.8709999968745,
  "completed_at": "2025-03-24 23:43:28.575",
  "units": {
    "inferred_brightnesses": "mpsas",
    "generated_in": "ms"
  },
  "inferred_brightnesses": {
    "8928308280fffff": 11.824697494506836
  },
  "is_night": false
}
```

### `POST /geojson`

> n.b. the geojson stored in the db is what the `snapshot` container uses to
> make brightness requests

Add a geojson object to the db.

```sh
# where `data.geojson` is some geojson file
curl -X POST -H "Content-Type: application/json" -d @data.geojson localhost:8000/geojson
```

### `GET /geojson`

Get the geojson objects from the db.

```sh
curl -X GET localhost:8000/geojson
```

```json
[
  {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": {},
        "geometry": {
          "coordinates": [
            [
              [
                -79.05384859000527,
                43.25389088469879
              ],
              [
                -79.05384859000527,
                42.48475162030496
              ],
              [
                -77.2358957097201,
                42.48475162030496
              ],
              [
                -77.2358957097201,
                43.25389088469879
              ],
              [
                -79.05384859000527,
                43.25389088469879
              ]
            ]
          ],
          "type": "Polygon"
        }
      }
    ]
  }
]
```
### `GET /version`

Get the hash and version of the sky brightness model.

```sh
curl -X GET localhost:8000/version
```

```json
{
  "hash": "9b1063ca1c578d67d33b2150ae1a8e5c9c7d87e1261373f2c507a2dd5fe37bf4",
  "version": "0.0.1"
}

```

## images

the `api` image is available from ghcr.io:

```sh
docker pull ghcr.io/nonnontrivial/ctts-api:latest

```

## licensing

This project is licensed under the AGPL-3.0 license.
