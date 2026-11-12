# syntax=docker/dockerfile:1
# API image. Default build is API-only with the `app` extra (no torch) and the
# deterministic hash embedder; see docker-compose.yml for the knobs.
#   docker build .                                      # API only
#   docker build --build-arg EXTRAS=app,ml .            # + torch (CPU), sentence-transformers, HF embeddings
#   docker build --build-arg EXTRAS=app,otel .          # + OpenTelemetry exporter
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
# Empty: torch is the CPU wheel pinned in requirements/ml.txt. docker-compose.gpu.yml sets a CUDA
# wheel index here; torch is then the same release from that index, and the CUDA packages it
# requires are resolved at build time (no lock file covers them; that build is by hand only).
ARG TORCH_INDEX_URL=
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app
WORKDIR /app

# Dependencies first (cached layer), from the lock (`make lock`): requirements/app.txt, then
# the pinned file of every other extra in EXTRAS (requirements/ml.txt, requirements/otel.txt),
# each with --no-deps, so nothing is resolved and nothing outside the files is installed.
# Every file is compiled under requirements.txt as a constraint, so a package two files share
# has one version; pip check confirms the files fit together. An extra with no pinned file
# stops the build.
COPY requirements/ ./requirements/
RUN set -eu; \
    pip install --no-deps -r requirements/app.txt; \
    for extra in $(echo "${EXTRAS}" | tr ',' ' '); do \
      if [ "${extra}" = app ]; then continue; fi; \
      if [ ! -f "requirements/${extra}.txt" ]; then \
        echo "EXTRAS=${EXTRAS}: no pinned file requirements/${extra}.txt (make lock writes one per extra)" >&2; exit 1; \
      fi; \
      if [ "${extra}" = ml ] && [ -n "${TORCH_INDEX_URL:-}" ]; then \
        grep -v '^torch==' requirements/ml.txt > "${TMPDIR:-/tmp}/ml-without-torch.txt"; \
        pip install --no-deps -r "${TMPDIR:-/tmp}/ml-without-torch.txt"; \
        pip install "torch==$(sed -n 's/^torch==\([^+]*\)+cpu$/\1/p' requirements/ml.txt)" --index-url "${TORCH_INDEX_URL}"; \
      else \
        pip install --no-deps -r "requirements/${extra}.txt"; \
      fi; \
    done; \
    pip check

# Source tree, installed in place so bench/ and app/ resolve from /app.
COPY . .
RUN pip install --no-deps -e .

COPY --from=frontend /frontend/out /app/static

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
