# CTTS

> closer to the stars

## how to build

### 1. write csv to disk

```sh
```

### 2. train pytorch model on saved csv

```sh
```

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
