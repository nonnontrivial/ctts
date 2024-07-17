FROM python:3.11.7-slim-bullseye

LABEL maintainer="Kevin Donahue <nonnontrivial@gmail.com>"

# system dependencies needed for psycopg3
RUN apt-get update \
    && apt-get install -y libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./requirements.txt .

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY . .

CMD python -m pc.consume
