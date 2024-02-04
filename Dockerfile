FROM python:3.11.7-slim-bullseye

LABEL maintainer="Kevin Donahue <nonnontrivial@gmail.com>"

WORKDIR /app
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
COPY . .

CMD python -m uvicorn ctts.api:app --host 0.0.0.0 --port 8000
