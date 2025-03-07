#!/bin/bash

volume_name="open-meteo-data"

sync_open_meteo_data() {
    # Download the digital elevation model
    docker run -it --rm -v open-meteo-data:/app/data ghcr.io/open-meteo/open-meteo sync copernicus_dem90 static

    # Download GFS temperature on 2 meter above ground
    docker run -it --rm -v open-meteo-data:/app/data ghcr.io/open-meteo/open-meteo sync ncep_gfs013 temperature_2m,cloud_cover --past-days 1

    # Check every 10 minutes for updated forecasts. Runs in background
    docker run -d --rm -v open-meteo-data:/app/data ghcr.io/open-meteo/open-meteo sync ncep_gfs013 temperature_2m,cloud_cover --past-days 1 --repeat-interval 10
}

if docker volume ls -q | grep -q "^${volume_name}$"; then
    echo "volume $volume_name exists; updating volume"
    sync_open_meteo_data
else
    echo "$volume_name does not exist; creating it"
    docker volume create --name open-meteo-data
    sync_open_meteo_data
fi
