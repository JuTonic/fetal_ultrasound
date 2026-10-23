FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg TORCH_HOME=/cache/torch UV_CACHE_DIR=/tmp/uv
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --extra cuda --no-dev --no-install-project
COPY src/ src/
RUN uv sync --locked --extra cuda --no-dev --no-editable
COPY configs/ configs/
COPY data/processed/ data/processed/
ENTRYPOINT ["/app/.venv/bin/python", "-m", "fetal.pipeline"]
