# ctts

> note to user: very much a work in progress

This is a [Go](https://go.dev/) application that records and predicts
[sky brightness](http://unihedron.com/projects/darksky/Instruction_sheet.pdf).
It uses [GCP](https://cloud.google.com/) and is designed to be configurable / tunable. 

## required env vars
```shell
# gcp cloud storage record name
CLOUD_STORAGE_CSV_FILENAME
# gcp app engine-specific
PROJECT_ID
LOCATION_ID
QUEUE_ID
```

### optional env vars
```shell
PORT
```

## develop
```shell
go run ./...
```