# syntax=docker/dockerfile:1
# API image. Default build is API-only with the `app` extra (no torch) and the
# deterministic hash embedder; see docker-compose.yml for the knobs.
#   docker build .                                      # API only
#   docker build --build-arg EXTRAS=app,ml .            # + torch, sentence-transformers, HF embeddings
#   docker build --build-arg WITH_FRONTEND=1 .          # + Next.js static export served at /
ARG WITH_FRONTEND=0

# ---- optional Next.js static export (next.config.json: output "export") ----
FROM node:20-slim AS frontend-1
WORKDIR /frontend
COPY .frontend/package.json .frontend/package-lock.json ./
RUN npm ci
COPY .frontend/ ./
RUN npm run build

FROM alpine:3.20 AS frontend-0
RUN mkdir -p /frontend/out

FROM frontend-${WITH_FRONTEND} AS frontend

# ---- API -------------------------------------------------------------------
FROM python:3.11-slim AS api
ARG EXTRAS=app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app
WORKDIR /app

# Dependencies first (cached layer), from the single source of truth.
COPY pyproject.toml scripts/print_deps.py ./scripts/
RUN mv scripts/pyproject.toml . \
    && python scripts/print_deps.py "${EXTRAS}" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# Source tree, installed in place so bench/ and app/ resolve from /app.
COPY . .
RUN pip install --no-deps -e .

COPY --from=frontend /frontend/out /app/static

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
