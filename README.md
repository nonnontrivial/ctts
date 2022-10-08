# ctts

> note to user: **very much** a work in progress

This is a [Go](https://go.dev/) application that records and predicts
[sky brightness](http://unihedron.com/projects/darksky/Instruction_sheet.pdf).
It uses [GCP](https://cloud.google.com/) and is designed to be configurable / tunable. 

It contains a small embedded [React](https://reactjs.org/) frontend.

## env vars
The following environment variables are requried to use ctts.
> note that a `.env` file can be used to load the following values directly.
```shell
PORT=
# gcp app engine-specific
GCP_PROJECT_ID=
GCP_DATASET_ID=
GCP_TABLE_ID=
GCP_LOCATION_ID=
GCP_QUEUE_ID=
```

## develop
```shell
go run ./...
```