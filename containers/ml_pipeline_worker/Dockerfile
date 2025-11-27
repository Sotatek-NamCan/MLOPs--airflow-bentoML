# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /opt/pipeline

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install .

ENTRYPOINT ["python"]
