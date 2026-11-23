FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[notebooks]"

COPY notebooks /notebooks
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
WORKDIR /notebooks
ENTRYPOINT ["/entrypoint.sh"]
