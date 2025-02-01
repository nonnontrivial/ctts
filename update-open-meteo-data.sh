#!/bin/bash

volume_name="open-meteo-data"

if docker volume ls -q | grep -q "^${volume_name}$"; then
    echo "volume $volume_name exists; updating volume"
    docker run -it --rm -v open-meteo-data:/app/data ghcr.io/open-meteo/open-meteo sync ecmwf_ifs04 cloud_cover,temperature_2m --past-days 3
else
    cmd="docker volume create --name open-meteo-data"
    echo "$volume_name does not exist; create it by running '$cmd'"
    exit 1
fi
