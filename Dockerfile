FROM python:3.12-slim

RUN \
    apt-get update && \
    apt-get install --yes pkg-config libhdf5-dev git &&  \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN \
    git clone https://github.com/gregversteeg/NPEET.git &&  \
    pip install ./NPEET && \
    rm -rf NPEET

RUN \
    apt-get remove --yes pkg-config libhdf5-dev git

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src src

RUN \
    mkdir data || true && \
    mkdir projects || true && \
    mkdir instance || true && \
    mkdir embeddings || true && \
    mkdir clustering || true && \
    mkdir dimensionality_reduction || true

ENV ENVIRONMENT_FILE="src/.env"

EXPOSE 1050

ENV PYTHONPATH=.

# TODO Is not reacheable
# ENTRYPOINT ["gunicorn", "--config", "src/gunicorn_config.py", "src.app:server"]
# TODO Is not reacheable
ENTRYPOINT ["python", "src/app.py"]
