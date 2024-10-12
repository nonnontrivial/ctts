ARG VERSION=3.12.6-slim-bookworm
FROM python:${VERSION}

LABEL maintainer="Kevin Donahue <nonnontrivial@gmail.com>"

WORKDIR /app
COPY ./requirements.txt .

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY . .

CMD python -m pc.main
