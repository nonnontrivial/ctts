# pc

> prediction consumer.

* pulls brightness observation messages off of `brightness.prediction` queue, inserting to postgres
* pulls messages pertaining to H3 cycle completeness off of `brightness.cycle` queue

```shell
# connect to postgres instance
psql -d "postgres://postgres:password@localhost/postgres"
```
