#!/bin/bash
# Pull required Ollama models on first start
# Runs automatically when the Ollama container starts
# Uses only the `ollama` CLI — no curl required

set -e

wait_for_ollama() {
    echo "Waiting for Ollama to be ready..."
    until ollama list > /dev/null 2>&1; do
        sleep 2
    done
    echo "Ollama is ready."
}

pull_if_missing() {
    local model="$1"
    echo "Checking model: $model"
    if ollama list 2>/dev/null | grep -q "^${model%:*}"; then
        echo "  ✓ Already available: $model"
    else
        echo "  ↓ Pulling: $model (this may take a while...)"
        ollama pull "$model"
        echo "  ✓ Pulled: $model"
    fi
}

wait_for_ollama

# ─── Required Models ──────────────────────────────────────────────────────────
# Simulation Agent: best code generation
pull_if_missing "qwen2.5-coder:32b"

# Database Agent: lighter code model
pull_if_missing "qwen2.5-coder:14b"

# Analytics Agent: strong reasoning
# Options depending on VRAM budget:
#   llama3.3:70b   ~40GB  - recommended with 2x RTX PRO 6000
#   deepseek-r1:32b ~20GB - alternative
pull_if_missing "llama3.3:70b"

# Optional: for orchestrator
# pull_if_missing "deepseek-r1:32b"

echo ""
echo "=== Model setup complete ==="
ollama list
