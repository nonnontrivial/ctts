# syntax=docker/dockerfile:1

FROM golang:1.19.2-bullseye

WORKDIR /app

COPY . .

RUN go build -o /ctts
EXPOSE 3303

CMD ["/ctts"]