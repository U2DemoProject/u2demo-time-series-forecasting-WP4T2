FROM python:3.12-slim

# Non-root user so files written to bind-mounted volumes (models/, output/, .cache/)
# don't land on the host as root. Callers can override at runtime with
# `--user "$(id -u):$(id -g)"` so files end up owned by the host user directly.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

ENTRYPOINT ["u2"]
