# Created by John Vorsten 2021-12-9
## Builder
FROM python:3.8-buster AS builder

# Working directory
WORKDIR /usr/src/app

# Build variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERRED 1

# Dependencies
WORKDIR /usr/src/app
COPY ./requirements-serve.txt /usr/src/app
RUN apt-get update && mkdir /usr/src/app/wheels && \
    pip wheel --wheel-dir /usr/src/app/wheels -r requirements-serve.txt

## Final
FROM python:3.8
RUN addgroup --system app_group && adduser --ingroup app_group app
WORKDIR /usr/src/app

# Networking
EXPOSE 8003

# Update and install dependencies
RUN apt update
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements-serve.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache /wheels/*

# Files
COPY ./compNB_si.clf .
COPY ./multiNB_si.clf .
COPY ./svmc_l1_miles.clf .
COPY ./svmc_l1_si.clf .
COPY ./svmc_rbf_miles.clf .
COPY ./svmc_rbf_si.clf .
COPY ./serving_config.ini .
COPY ./*.py .
COPY ./miles_concept_features.dat .
COPY ./typescorrection.csv .
COPY ./categorical_categories.dat .
COPY ./vocab_name.txt .
COPY ./vocab_descriptor.txt .

# Runtime user
USER app

# Entrypoint & Daemon
ENTRYPOINT ["uvicorn"]
CMD ["--host", "0.0.0.0", "--port","8003", "mil_serving:app","--access-log","--log-level","info"]