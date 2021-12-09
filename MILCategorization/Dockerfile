# Created by John Vorsten 2021-12-9
## Builder
FROM python:3.8-buster as builder

# Working directory
WORKDIR /usr/src/app

# Build variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERRED 1

# Dependencies
WORKDIR /usr/src/app
COPY ./requirements-serve.txt
RUN apt-get update && mkdir /usr/src/app/wheels && \
    pip instll --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements-serve.txt

## Final
FROM python:3.8-buster
RUN addgroup --system app_group && adduser --ingroup app_group app
WORKDIR /usr/src/app

# Networking
EXPOSE 8003

# Update and install dependencies
RUN apt update
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements-serve.txt .
RUN pip install --upgrade pip && \
    pip install --nop-cache /wheels/*

# Files
copy ./*.clf .
COPY ./serving_config.ini .
COPY ./*.py .
COPY ./miles_concept_features.dat .

# Runtime user
USER app

# Entrypoint & Daemon
#TODO Logfile
#TODO bind ports in run command
CMD ["uvicorn","--port","8003", "mil_serving:app","--access-log","--log-level","info"]