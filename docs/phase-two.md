## phase two: _APIs / containerization_

The second phase should be to expose predictions over HTTP in a containerized way.

The reason we want to do this is twofold: support experimentation with different
frontends in the next phase, and support testing and verification of the predict
capability.

### requirements

- can GET /predict, where query params are
  - `lat`: latitude
  - `lon`: longitude
  - `astro_twilight_type`: one of `next` | `previous` | `nearest`, denoting which astronomical twilight should be computed relative to [`Time.now`](https://docs.astropy.org/en/stable/time/#getting-the-current-time) at point of endpoint receiving the request
- can `docker run` the project
