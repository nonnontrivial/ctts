# syntax=docker/dockerfile:1
FROM golang:1.16-alpine3.13

WORKDIR /app

COPY go.mod ./
COPY go.sum ./
RUN go mod download

COPY . ./

RUN go build -o /ctts

EXPOSE 8080

CMD ["/ctts"]