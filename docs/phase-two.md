## phase two: _APIs / containerization_

The second phase should be to expose predictions over HTTP in a containerized way.

The reason we want to do this is twofold: support experimentation with different
frontends in the next phase, and support testing and verification of the predict
capability.

### requirements

Making a GET request to the `/predict` endpoint, e.g.

```sh
curl http://localhost:5000/api/predict?lat=42.0&lon=42.0&astro_twilight_type=nearest
```

(where `astro_twilight_type` can be `next` | `previous` | `nearest`, denoting which astronomical twilight should be computed relative to the [`Time.now`](https://docs.astropy.org/en/stable/time/#getting-the-current-time))

should return a result like

```json
{
  "y": 22.0,
  "astro_twilight": "2024-01-01 23:28:05 UTC",
  "model": {
    "version": "0.0.1",
    "loss": 0.1
  }
}
```
