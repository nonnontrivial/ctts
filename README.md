# CTTS

## purpose

> The purpose of CTTS is to develop APIs that bring the night sky closer to the user.

## sky brightness model

CTTS contains a _sky brightness model_, which can predict the sky brightness of a site. A site is a latitude and longitude at a date and time.

### running locally

> Note: tested on python 3.11

```sh
cd ctts
pip install -r requirements.txt
python -m uvicorn prediction.api:app --reload
```

```sh
curl "http://localhost:8000/api/v1/prediction?lat=-30.2466&lon=-70.7494&astro_twilight_type=next"

```

```json
{
	"sky_brightness": 22.0388,
	"astronomical_twilight_iso": "2024-01-11 01:23:49.216"
}
```

### endpoints

#### _`/api/v1/prediction`_

Gets the predicted sky brightness at (default) nearest [astronomical twilight](https://www.weather.gov/lmk/twilight-types#:~:text=Astronomical%20Twilight%3A,urban%20or%20suburban%20light%20pollution.) to provided `lat` and `lon`.

Optional query param `astro_twilight_type` can be `nearest` | `next` | `previous` to denote the astronomical twilight that should be used relative to the [Time.now](https://docs.astropy.org/en/stable/api/astropy.time.Time.html#astropy.time.Time.now).

#### swagger ui

Open the [ui](http://localhost:8000/docs) in a browser.

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

### validation

> Note: validation of the model's predictions is an ongoing process.

### running tests

```sh
cd ctts
python -m pytest
```

### env

Location of logfile can be controlled by setting `SKY_BRIGHTNESS_LOGFILE` to
some filename (in the home directory).

```sh
export SKY_BRIGHTNESS_LOGFILE=ctts.log
```
