#!/bin/bash
#SBATCH --job-name=metanavit-setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=setup_models_%j.log

# MetaNaviT Model Setup Script
# Usage:
#   Interactive: srun --partition=gpu --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash scripts/setup_models.sh
#   Batch:       sbatch scripts/setup_models.sh

set -e

echo "=== MetaNaviT Model Setup ==="
echo "Host: $(hostname)"
echo "Date: $(date)"

# Check GPU access
if nvidia-smi -L 2>/dev/null; then
    echo "[OK] GPUs detected"
else
    echo "[ERROR] No GPU detected. Run this on a GPU node:"
    echo "  srun --partition=gpu --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash scripts/setup_models.sh"
    exit 1
fi
echo ""

# Load modules
module load cuda/12.4 2>/dev/null || true

# Use user-local Ollama if available, otherwise HPC module, otherwise install
OLLAMA=""
if [ -x "$HOME/.local/bin/ollama" ]; then
    OLLAMA="$HOME/.local/bin/ollama"
elif command -v ollama &>/dev/null; then
    OLLAMA="ollama"
else
    echo "[INSTALL] Downloading Ollama to ~/.local/bin ..."
    mkdir -p ~/.local/bin
    LATEST=$(curl -sL https://api.github.com/repos/ollama/ollama/releases/latest | grep tag_name | cut -d'"' -f4)
    gh release download "$LATEST" -R ollama/ollama -p "ollama-linux-amd64.tar.zst" -D /tmp/ --clobber
    zstd -d /tmp/ollama-linux-amd64.tar.zst -o /tmp/ollama-linux-amd64.tar 2>/dev/null
    tar -xf /tmp/ollama-linux-amd64.tar -C ~/.local/
    chmod +x ~/.local/bin/ollama
    OLLAMA="$HOME/.local/bin/ollama"
    rm -f /tmp/ollama-linux-amd64.tar.zst /tmp/ollama-linux-amd64.tar
fi

echo "[OK] Using Ollama: $($OLLAMA --version 2>&1 | head -1)"

# Start Ollama server if not running
export OLLAMA_MODELS="$HOME/.ollama/models"
if curl -sf http://localhost:11434/api/version &>/dev/null; then
    echo "[OK] Ollama server already running"
else
    echo "[START] Starting Ollama server..."
    OLLAMA_HOST=http://127.0.0.1:11434 $OLLAMA serve &>/tmp/ollama-serve.log &
    OLLAMA_PID=$!
    sleep 5
    if curl -sf http://localhost:11434/api/version &>/dev/null; then
        echo "[OK] Ollama server started (PID $OLLAMA_PID)"
    else
        echo "[ERROR] Failed to start Ollama server. Log:"
        cat /tmp/ollama-serve.log
        exit 1
    fi
fi

# Pull LLM model
echo ""
echo "[PULL] Pulling qwen2.5:14b (~9GB)..."
OLLAMA_HOST=http://localhost:11434 $OLLAMA pull qwen2.5:14b

# Verify
echo ""
echo "=== Installed Models ==="
OLLAMA_HOST=http://localhost:11434 $OLLAMA list
echo ""
echo "=== Setup Complete ==="
echo "Ollama server running at http://localhost:11434"
echo ""
echo "Next steps:"
echo "  cd $(cd "$(dirname "$0")/.." && pwd)"
echo "  ./scripts/run.sh generate   # build embeddings"
echo "  ./scripts/run.sh dev         # start the app"
