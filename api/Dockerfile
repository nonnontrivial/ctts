ARG VERSION=3.12.6-slim-bookworm
FROM python:${VERSION}

LABEL maintainer="Kevin Donahue <nonnontrivial@gmail.com>"

RUN pip install uv

WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY . .

CMD python -m api.main
