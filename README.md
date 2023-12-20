# CTTS

## purpose

The purpose of CTTS is to bring the night sky closer to the user.

For now, this will take the form of notifying the user when a site that they are
watching has a predicted brightness (at that site's astronomical twilight) above
a threshold they set.

## phase1

The first phase of this is to get the data and contruct the model:

- ability to find nearest astronomical twilight for a lat,lon,datetime
- definition of runtime-constructable features expected to impact sky brightness
- ability to clean/aggregate data containing Xs -> y relationship and form model

## phase2

Now that we have a model producing outputs that make sense, there needs to be a
daily cron jon that:

- determines which sites have brightness predicted to be above their threshold
- publishes message containing info for sites above threshold
