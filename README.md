# ctts

> note to user: **very much** a work in progress

This is a [Go](https://go.dev/) application that records and predicts
[sky brightness](http://unihedron.com/projects/darksky/Instruction_sheet.pdf).
It uses [GCP](https://cloud.google.com/) and is designed to be configurable / tunable. 

It contains a small embedded [React](https://reactjs.org/) frontend.

## env vars
### required
```shell
# gcp cloud storage record name
CLOUD_STORAGE_CSV_FILENAME
# gcp app engine-specific
GCP_PROJECT_ID
GCP_LOCATION_ID
GCP_QUEUE_ID
```

### optional
```shell
PORT
```

## develop
```shell
go run ./...
```