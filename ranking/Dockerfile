# Created by John Vorsten 2021-12-9
## Builder
FROM python:3.7-slim AS builder

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
FROM python:3.7-slim
RUN addgroup --system app_group && adduser --ingroup app_group app
WORKDIR /usr/src/app

# Networking
EXPOSE 8004

# Update and install dependencies
RUN apt update
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements-serve.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache /wheels/*

# Files
COPY ./default_serving_peritem_features.dat .
COPY ./model4_serving.py .
COPY ./model4_serving_test.py .
COPY ./ranking_model_4.py .
COPY ./serving_config.ini .

# Runtime user
USER app

# Entrypoint & Daemon
ENTRYPOINT ["uvicorn"]
CMD ["--host", "0.0.0.0", "--port","8004", "model4_serving:app","--access-log","--log-level","info"]