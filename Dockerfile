FROM python:3.11-slim

RUN apt-get update && apt-get install -y pkg-config libhdf5-dev git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /

RUN pip install --no-cache-dir -r /requirements.txt


RUN \
    git clone https://github.com/gregversteeg/NPEET.git &&  \
    pip install ./NPEET && \
    rm -rf NPEET

WORKDIR /

COPY src/ ./

RUN mkdir -p ./instance

ENV ENVIRONMENT_FILE=".env"

EXPOSE 1050

ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "app:server"]
