FROM python:3.10-slim

# Install poetry
RUN pip install poetry

# Setup dependencies
WORKDIR /app
COPY README.md pyproject.toml poetry.lock /app/

COPY koregraph/ .

RUN poetry config virtualenvs.create false
RUN poetry install --with training --no-root --no-interaction --no-ansi

CMD ["poetry", "run", "mlflow", "server"]
