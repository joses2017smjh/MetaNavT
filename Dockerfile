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
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app
WORKDIR /app

# Dependencies first (cached layer), from the lock: requirements/app.txt is the pinned
# `app` extra written by `make lock`, installed exactly as committed. The default build
# (EXTRAS=app) resolves nothing else.
# Extras with no pinned file (ml, otel) are still ranges from pyproject.toml; they are
# installed afterwards with the pinned file as a constraint, so they can add packages
# but cannot move a pinned one. Torch comes from the selected wheel index before the
# ml extra. CPU remains the default; docker-compose.gpu.yml explicitly selects CUDA wheels.
COPY pyproject.toml scripts/print_deps.py requirements/app.txt ./scripts/
RUN mv scripts/pyproject.toml . \
    && pip install -r scripts/app.txt \
    && if [ "${EXTRAS}" != "app" ]; then \
         (case ",${EXTRAS}," in *,ml,*) pip install torch --index-url "${TORCH_INDEX_URL}" ;; esac) \
         && python scripts/print_deps.py "${EXTRAS}" > /tmp/extras.txt \
         && pip install -c scripts/app.txt -r /tmp/extras.txt ; \
       fi

# Source tree, installed in place so bench/ and app/ resolve from /app.
COPY . .
RUN pip install --no-deps -e .

COPY --from=frontend /frontend/out /app/static

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
