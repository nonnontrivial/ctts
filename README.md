# CTTS

## purpose

> The purpose of CTTS is to bring the night sky closer to the user.

For now, this will take the form of _notifying the user when a site that they are
watching has a predicted brightness at that site's astronomical twilight which is above
a threshold the user sets_.

## run

> Note: messaging works for macOS only

> Note: tested on python3.11

```sh
cd ctts
pip install -r requirements.txt
python -m prediction.predict
```

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
