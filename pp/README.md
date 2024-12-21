# pp

> prediction producer.

Retrieves sky brightness prediction across in-polygon h3 cells and puts results on rabbitmq.

> in-polygon is the interior of `land.geojson`

## monitoring

see the rabbitmq [dashboard](http://localhost:15672/#/)

## running the tests

```shell
python3 -m pytest
```
