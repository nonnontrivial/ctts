# notebooks

## purpose

The purpose of these notebooks is to develop a model that can predict sky brightness at
a location's nearest astronomical twilight.

## requirements

To do that, we need a few things:

- definition of runtime-constructable features that are expected to impact sky brightness
- ability to clean/aggregate datasets with relationship between constructable features and y (sqm reading in mpsas) into single dataframe
- construction of model relating constructable features to sky brightness (in mpsas)
- loss of model (on test data) low enough to allow for differentiation between upper bortle classes (<0.5)
- ability to find nearest astronomical twilight for a lat,lon,datetime
- script for constructing features and using saved model's predicition on those features at nearest astro twilight
