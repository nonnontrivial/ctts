# CTTS

## purpose

> The purpose of CTTS is to bring the night sky closer to the user.

For now, this will take the form of _notifying the user when a site that they are
watching has a predicted brightness at that site's astronomical twilight which is above
a threshold the user sets_.

## run

> Note: messaging works for macOS only

```sh
cd ctts
python -m prediction.predict
```

## phase one: modeling & verification

The first phase of this is to get the data, construct the model, and then prove out
the results..

### requirements

- ability to find nearest astronomical twilight for a lat,lon,datetime
- definition of runtime-constructable features expected to impact sky brightness
- ability to clean/aggregate data containing Xs -> y relationship and form model
- ability to create new Xs on data around a lat,lon,datetime
- ability to persist predicted result (logfile) and compare with ground truth (..?)
- ability to run daily as a launchd service

## phase two: messaging

Now that there are reasonable predictions coming out of the model, the user should
be notified when a prediction is at or above that site's threshold..

### requirements

- ability to generate site summaries for each user
- ability to send an iMessage to each user

## phase three: infrastructure & data storage
