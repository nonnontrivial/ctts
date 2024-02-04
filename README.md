# CTTS

> closer to the stars

## [HTTP APIs](#api.md)

- artificial sky brightness (light pollution)
- predictive sky brightness

### running locally

> Note: tested on python 3.11

```sh
cd ctts
pip install -r requirements.txt
python -m uvicorn ctts.api:app --reload
```

### with docker

```sh
cd ctts
docker-compose up -d
```

### running tests

```sh
cd ctts
python -m pytest
```
