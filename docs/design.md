# design

## phase one: _modeling / verification / messaging_

- build the model on slightly augmented GaN dataset
- verify that model predictions make some intuitive sense
- message a "user" when prediction at nearest astronomical twilight is above threshold in csv

### requirements

- ability to find nearest astronomical twilight for a lat,lon,datetime
- definition of runtime-constructable features expected to impact sky brightness
- ability to clean/aggregate data containing Xs -> y relationship and form model
- ability to create new Xs on data around a lat,lon,datetime
- ability to persist predicted result (logfile) and compare with ground truth (..?)
- ability to run daily as a launchd service
- ability to generate site summaries for each "user"
- ability to send an iMessage to each "user"

### run

> Note: messaging works for _macOS only_ due to use of iMessage and Shortcuts

> Note: tested on python3.11

```sh
cd ctts
git checkout v0.1.0
pip install -r requirements.txt
```

```sh
python -m prediction.predict
```

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
