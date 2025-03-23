# consumers

## `store_in_sqlite.py`

> n.b. assumes main stack is running (with `docker compose up`)

- stores each snapshot in the main db file `./data/ctts.db`
- generates folium maps for each snapshot (with brightness values mapped to hex color values)

```sh
uv run store_in_sqlite.py
```
