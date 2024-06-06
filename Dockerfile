# FROM python:3.10-slim

# # Install poetry
# RUN pip install poetry

# # Setup dependencies

# COPY koregraph/ .

# RUN poetry config virtualenvs.create false
# RUN poetry install --with training --no-root --no-interaction --no-ansi

# EXPOSE 5000

# CMD ["poetry", "run", "mlflow", "server", "--host", "0.0.0.0"]

FROM python:3.10-slim

RUN pip install mlflow
# RUN poetry config virtualenvs.create false

# WORKDIR /app
# COPY README.md pyproject.toml poetry.lock /app/

# RUN poetry add mlflow
# RUN poetry install

EXPOSE 8000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8000"]
