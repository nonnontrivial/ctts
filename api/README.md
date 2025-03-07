# api

API server for sky brightness in terms of H3 cells.

## run

Startup the API server:

```sh
uv run fastapi dev
```

Make a request:

```sh
curl -X POST -H 'Content-Type: application/json' -d '["8928308280fffff"]' localhost:8000/infer
```
