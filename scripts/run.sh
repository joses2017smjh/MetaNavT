#!/bin/bash
# Convenience wrapper. Activate whatever Python environment has the `app` extra
# first (pip install -e ".[eval,app]"); no conda required.
set -e
cd "$(dirname "$0")/.."

case $1 in
  "generate")
    python -m app.database.migrate   # the tables and indexes come from the schema migrations
    python -m app.engine.generate    # the index step (same as make index): DATA_DIR into pgvector, only changed files embedded
    ;;
  "dev")
    python run.py dev
    ;;
  "build")
    python run.py build
    ;;
  *)
    echo "Usage: ./scripts/run.sh [generate|dev|build]"
    exit 1
    ;;
esac
