# ChurnGuard scoring API. Data, registry DB and model artifacts are mounted, not baked in:
#   make docker-build && make docker-run
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PYTHONUNBUFFERED=1

# LightGBM needs the OpenMP runtime, which python:*-slim does not ship. Required even with a
# logreg champion: unpickling imports the candidates module, which imports lightgbm.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 1000 churnguard && chown -R churnguard /app
USER churnguard

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uv", "run", "--no-sync", "uvicorn", "churnguard.serving.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
