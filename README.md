# CTTS

## getting started

> Note: tested on python 3.11

### running the fastAPI server

> Note: you may have to [build the training data](#training-guide.md) before you
> can get results from the prediction endpoint.

```sh
cd ctts
pip install -r requirements.txt
python -m uvicorn ctts.api:app --reload
```

#### with docker

```sh
cd ctts
docker-compose up -d
```

#### running tests

```sh
cd ctts
python -m pytest
```
