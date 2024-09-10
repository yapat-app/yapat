FROM python:3.11-slim

COPY requirements.txt /

RUN apt-get update && apt-get install -y pkg-config libhdf5-dev && rm -rf /var/lib/apt/lists/* \

RUN pip install -r /requirements.txt && rm -rf /root/.cache

COPY src/ ./

RUN mkdir -p ./data ./projects ./instance

ENV ENVIRONMENT_FILE=".env"

EXPOSE 1050

ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "app:server"]
