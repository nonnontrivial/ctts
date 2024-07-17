# pc

prediction consumer.

pulls predictions messages off of prediction queue and into postgres and websockets. 

## connect to timescale instance

```shell
psql -d "postgres://postgres:password@localhost/postgres"
```
