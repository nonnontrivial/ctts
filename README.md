# CTTS

> Closer To The Stars

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
