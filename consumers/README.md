# consumers

## `store_in_sqlite.py`

> n.b. assumes main stack is running (with `docker compose up`)

- stores each snapshot in a SQLite database (`brightness.db`)
- generates folium maps for each snapshot (with brightness values mapped to hex color values)

```sh
uv run store_in_sqlite.py
```
