# CTTS

## purpose

The purpose of CTTS is to bring the night sky closer to the user.

For now, this will take the form of notifying the user when a site that they are
watching has a predicted brightness (at that site's astronomical twilight) above
a threshold they set.

## phase1

The first phase of this is to get the data and contruct the model:

- ability to find nearest astronomical twilight for a lat,lon,datetime
- definition of runtime-constructable features that are expected to impact sky brightness
- ability to clean/aggregate data containing Xs -> y relationship and form model

## phase2

Now that we have a model producing outputs that make sense, we need a cron task
that can understand when a site's prediction should generate a notification:

- ability to store sites somewhere accessible (cloud sql)
- ability to run prediction on sites on a schedule (cloud function + cloud scheduler)
- ability to send a SMS (or imessage?) text to a number

## phase3
