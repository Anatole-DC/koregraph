FROM python:3.10-slim

# Install poetry
RUN pip install poetry

# Setup dependencies
WORKDIR /app
COPY README.md pyproject.toml poetry.lock /app/

COPY python_template/ .

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi
