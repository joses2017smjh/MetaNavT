#!/bin/bash
# Convenience wrapper. Activate whatever Python environment has the `app` extra
# first (pip install -e ".[eval,app]"); no conda required.
set -e
cd "$(dirname "$0")/.."

case $1 in
  "generate")
    python -m app.engine.generate
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
