# CTTS

## purpose

> The purpose of CTTS is to develop APIs that bring the night sky closer to the user.

## sky brightness model

CTTS contains a sky brightness model, which allows getting a sky brightness prediction at nearest astronomical twilight to a latitude and longitude.

### running with docker

- build the image

```sh
cd ctts
docker build -t ctts:latest .
```

- run the container

```sh
docker run -d --name ctts -p 8000:80 ctts:latest
```

- GET `/predict` to see model's predicted sky brightness at a `lat`, `lon` for that site's astronomical twilight.

```sh
curl "http://localhost:8000/api/predict?lat=-30.2466&lon=-70.7494"
```

```json
{
  "brightness_mpsas": 19.9871,
  "astro_twilight": { "iso": "2023-12-28 01:23:32.453 UTC", "type": "nearest" }
}
```
