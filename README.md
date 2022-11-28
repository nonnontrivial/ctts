# ctts

> note to user: **very much** a work in progress

This is a [Go](https://go.dev/) application that records and predicts
[sky brightness](http://unihedron.com/projects/darksky/Instruction_sheet.pdf).

It contains a small embedded [React](https://reactjs.org/) frontend.

## env vars

The following environment variables are requried to use ctts.

> note that a `.env` file can be used to load the following values directly.

```shell
PORT=
```

## develop

Generating the server gRPC code happens like this:

> note: need to install the grpc web binary first

```shell
protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative proto/service/service.proto
```

Generating the client gRPC code happens like this:

```shell
protoc -I=proto service/service.proto \
  --js_out=import_style=commonjs,binary:client/src/grpc \
  --grpc-web_out=import_style=typescript,mode=grpcweb:client/src/grpc
```

```shell
go run *.go
```
