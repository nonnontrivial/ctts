# CTTS

APIs that tap into the quality of the night sky.

## HTTP APIs

- artificial sky brightness (light pollution)
- predictive sky brightness

### running locally

> Note: tested on python 3.11

```sh
pip install -r requirements.txt
python -m uvicorn ctts.api:app --reload
```

### endpoints

#### `/api/v1/pollution`

Gets the approximate artificial sky brightness map [RGBA pixel value](https://djlorenz.github.io/astronomy/lp2022/colors.html) for a lat and lon (for the year 2022).

```sh
curl "localhost:8000/api/v1/pollution?lat=40.7277478&lon=-74.0000374"
```

```json
{"r":255,"g":255,"b":255,"a":255}
```

#### `/api/v1/prediction`

Gets the predicted sky brightness at `lat` and `lon`.

```sh
curl "http://localhost:8000/api/v1/prediction?lat=-30.2466&lon=-70.7494"

```

```json
{
	"sky_brightness": 22.0388
}
```


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
