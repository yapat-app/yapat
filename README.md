# YAPAT
## Yet another PAM Annotation Tool

## Clone repository

```bash
git clone --branch dev https://github.com/yapat-app/yapat.git
cd yapat
```

> **NOTE**: All following commands assume the same directory.

## Create a virtual environment

```bash
python3.12 -m venv venv
```

## Upgrade package manager

```bash
venv/bin/python -m pip install --upgrade pip
```

## Install dependencies

```bash
pip install yapat
```

## Run the application

```bash
PORT=1050 PYTHONPATH=. venv/bin/python src/app.py
```

## Open the application

-   Open <http://localhost:1050>
