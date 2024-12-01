FROM python:3.11-slim

COPY requirements.txt /

RUN pip install --no-cache-dir -r /requirements.txt

RUN pip install -r /requirements.txt && rm -rf /root/.cache

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
