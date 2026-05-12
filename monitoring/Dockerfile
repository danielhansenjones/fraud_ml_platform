FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project --no-dev --frozen

COPY monitoring/ ./monitoring/

# Match the serving and router containers; nothing here needs root.
RUN useradd -r -u 1001 appuser && chown -R appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# Each compose service overrides `command`; this is just a friendly default
# for ad-hoc `docker run` so the container shows what it can do instead of
# silently exiting.
CMD ["python", "monitoring/main.py", "--help"]