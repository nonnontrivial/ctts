#!/bin/bash

volume_name="open-meteo-data"

sync_open_meteo_data() {
    docker run -it --rm -v open-meteo-data:/app/data ghcr.io/open-meteo/open-meteo sync ecmwf_ifs04 cloud_cover,temperature_2m --past-days 3
}

if docker volume ls -q | grep -q "^${volume_name}$"; then
    echo "volume $volume_name exists; updating volume"
    sync_open_meteo_data
else
    echo "$volume_name does not exist; creating it"
    docker volume create --name open-meteo-data
    sync_open_meteo_data
fi
