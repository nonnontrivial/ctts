# CTTS

> n.b.: this is _alpha software_; apis may change quickly, and quality of the brightness prediction is still being ironed out

CTTS is an open source application for reading [sky brightness](https://en.wikipedia.org/wiki/Sky_brightness) without a sensor.

## features

- api server for sky brightness at given H3 cells
- continuous "snapshots" of sky brightness over H3 cells in given geojson file

## run

To continuously generate snapshots of sky brightness over H3 cells in given geojson file:

1. clone the repo
2. sync openmeteo data: `./sync-open-meteo-data.sh`
3. create `./data.geojson` (cells will be made from the exterior space of polygons in this file)
4. run the containers: `docker compose up --build`

> logs in docker should look like this:

```log
api-1        |       INFO   172.18.0.5:53506 - "POST /infer HTTP/1.1" 200
snapshot-1   | 2025-03-08 15:45:26,085 - INFO - HTTP Request: POST http://api/infer "HTTP/1.1 200 OK"
snapshot-1   | 2025-03-08 15:45:26,089 - INFO - published data for 30 cells to brightness.snapshot
```

5. run `consume.py` in `scripts/` to confirm brightness messages are published to the queue

```log
```

## licensing

This project is licensed under the AGPL-3.0 license.
