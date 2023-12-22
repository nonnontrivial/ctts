#!/bin/bash
# Runs the prediction on the known sites.
# Intended to be run from launchd.
script_dir=$(dirname "$(readlink -f "$0")")
pip3 install -r "$script_dir/requirements.txt"

script_path="$script_dir/get_sky_brightness_at_known_sites.py"
python3 "$script_path"