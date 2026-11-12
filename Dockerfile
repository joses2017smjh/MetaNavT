# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# API image. Default build is API-only with the `app` extra (no torch) and the
# deterministic hash embedder; see docker-compose.yml for the knobs.
#   docker build .                                      # API only
#   docker build --build-arg EXTRAS=app,ml .            # + torch (CPU), sentence-transformers, HF embeddings
#   docker build --build-arg EXTRAS=app,otel .          # + OpenTelemetry exporter
#   docker build --build-arg WITH_FRONTEND=1 .          # + Next.js static export served at /
# Every image here, in docker-compose.yml and in CI is pinned by digest; the tag in front of each
# digest is the one it was looked up from, and the comment beside it the release that tag named
# then (2026-11-11). To move to newer images: doc/engineering-reference.md, Pinned base images.
ARG WITH_FRONTEND=0

# ---- optional Next.js static export (next.config.json: output "export") ----
# node:20-slim = 20.20.2-slim
FROM node:20-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0 AS frontend-1
WORKDIR /frontend
COPY .frontend/package.json .frontend/package-lock.json ./
RUN npm ci
COPY .frontend/ ./
RUN npm run build

# alpine:3.20 = 3.20.10
FROM alpine:3.20@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc AS frontend-0
RUN mkdir -p /frontend/out

FROM frontend-${WITH_FRONTEND} AS frontend

# ---- API -------------------------------------------------------------------
# python:3.11-slim = 3.11.18-slim (Debian trixie; it ships setuptools 79.0.1, which requirements/app.txt replaces)
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS api
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
