## phase one: _modeling / verification / messaging_

The first phase of this is to model sky brightness and verify that the model's
predictions make sense (while also testing a hypothetical workflow whereby a
"user" is messaged when a site they watch is predicted to have a "good" sky
brightness value at its nearest astronomical twilight).

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

> Note: messaging works for macOS only due to use of iMessage and Shortcuts

> Note: tested on python3.11

```sh
cd ctts
git checkout v0.1.0
pip install -r requirements.txt
```

```sh
python -m prediction.predict
```
