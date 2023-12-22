# CTTS

## purpose

The purpose of CTTS is to bring the night sky closer to the user.

For now, this will take the form of _notifying the user when a site that they are
watching has a predicted brightness at that site's astronomical twilight which is above
a threshold they set_.

## phase one: modeling & verification

The first phase of this is to get the data and contruct the model, and prove out
the results.

### requirements

- ability to find nearest astronomical twilight for a lat,lon,datetime
- definition of runtime-constructable features expected to impact sky brightness
- ability to clean/aggregate data containing Xs -> y relationship and form model
- ability to create new X on data around a lat,lon,datetime
- ability to persist predited result and compare with ground truth
- ability to run as a launchd service
