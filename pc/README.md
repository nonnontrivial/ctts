# pc

> prediction consumer.

Pulls brightness observation messages off of the prediction queue and:

- inserts to postgres
- broadcasts over websockets connection

```shell
# connect to postgres instance
psql -d "postgres://postgres:password@localhost/postgres"
```
