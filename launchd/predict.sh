#!/bin/bash
# Runs the prediction on the known sites.
# Intended to be run from launchd.

script_dir=$(dirname "$(readlink -f "$0")")
cd "$script_dir" || exit 1

/usr/local/bin/pip3 install -r "requirements.txt"
/usr/local/bin/python3 -m "prediction.predict"