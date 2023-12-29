# CTTS

## purpose

> The purpose of CTTS is to develop APIs that bring the night sky closer to the user.

## sky brightness model

CTTS contains a _sky brightness model_, which can predict the sky brightness of a site.

### running locally

> Note: tested on python 3.11

```sh
cd ctts
pip install -r requirements.txt
python -m uvicorn prediction.api:app --reload
```

```sh
curl "http://localhost:8000/api/prediction?lat=-30.2466&lon=-70.7494"
```

```json
{
  "brightness_mpsas": 19.9871,
  "astro_twilight": { "iso": "2023-12-28 01:23:32.453 UTC", "type": "nearest" }
}
```

### endpoints

#### _`/api/prediction`_

Gets the predicted sky brightness at nearest [astronomical twilight](https://www.weather.gov/lmk/twilight-types#:~:text=Astronomical%20Twilight%3A,urban%20or%20suburban%20light%20pollution.) to provided `lat` and `lon`.

### running with docker

> Note: image size is on the order of 5.13GB

- build the image

```sh
cd ctts
docker build -t ctts:latest .
```

- run the container

```sh
docker run -d --name ctts -p 8000:80 ctts:latest
```

### running tests

```sh
cd ctts
python -m pytest
```
