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

COPY src/ ./src

RUN mkdir -p ./instance
RUN mkdir -p ./embeddings
RUN mkdir -p ./clustering
RUN mkdir -p ./dimensionality_reduction

ENV ENVIRONMENT_FILE="src/.env"

EXPOSE 1050

COPY main.py .

ENV PYTHONPATH=.

# Reachable from outside the container.
ENTRYPOINT ["python", "main.py"]
# ENTRYPOINT ["gunicorn", "main:main"]  # Not reachable from outside the container.
# ENTRYPOINT ["gunicorn", "--config", "src/gunicorn_config.py", "main:main"]  # Not reachable from outside the container.

# ENTRYPOINT ["python", "src/app.py"]  # Not reachable from outside the container.
# ENTRYPOINT ["gunicorn", "src.app:main"]  # Not reachable from outside the container.
# ENTRYPOINT ["gunicorn", "--config", "src/gunicorn_config.py", "src/app:main"]  # Not reachable from outside the container.
