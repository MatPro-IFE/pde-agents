#!/bin/bash
# PDE Agents - One-time Setup Script
# Run this once to initialize everything

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail() { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         PDE Agents Setup                                ║"
echo "║  FEniCSx + LangGraph + Ollama + PostgreSQL + Dash       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Check prerequisites ──────────────────────────────────────────────────────
log "Checking prerequisites..."
command -v docker     || fail "Docker not found. Install from https://docs.docker.com/engine/install/"
command -v docker     && log "  ✓ Docker: $(docker --version)"

if ! docker info > /dev/null 2>&1; then
    fail "Docker daemon not running. Start it with: sudo systemctl start docker"
fi

# Check NVIDIA GPU
if nvidia-smi > /dev/null 2>&1; then
    log "  ✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        log "    $line"
    done
else
    warn "  NVIDIA GPU not detected. Some services may run slower."
fi

# Check for NVIDIA Container Toolkit
if docker info 2>/dev/null | grep -q "nvidia"; then
    log "  ✓ NVIDIA Container Toolkit: OK"
elif command -v nvidia-ctk > /dev/null 2>&1; then
    log "  ✓ NVIDIA Container Toolkit: OK (nvidia-ctk found)"
else
    warn "  NVIDIA Container Toolkit may not be installed."
    warn "  Install with:"
    warn "    sudo apt-get install -y nvidia-container-toolkit"
    warn "    sudo nvidia-ctk runtime configure --runtime=docker"
    warn "    sudo systemctl restart docker"
fi

# ─── Create .env file ─────────────────────────────────────────────────────────
log "Setting up environment file..."
if [ ! -f ".env" ]; then
    cp env.example .env
    log "  Created .env from env.example (edit to customize)"
else
    log "  .env already exists, skipping"
fi

# ─── Create result directories ───────────────────────────────────────────────
log "Creating result directories..."
mkdir -p results meshes mlflow_data
chmod 755 results meshes mlflow_data
log "  Created: results/, meshes/, mlflow_data/"

# ─── Make scripts executable ──────────────────────────────────────────────────
chmod +x scripts/ollama-init.sh
log "  Made scripts executable"

# ─── Pull base Docker images ──────────────────────────────────────────────────
log "Pulling Docker images (this may take a while on first run)..."
docker compose pull --quiet 2>/dev/null || warn "Some images failed to pull (will build locally)"

# ─── Build custom images ──────────────────────────────────────────────────────
log "Building custom Docker images..."
docker compose build --quiet
log "  ✓ Images built"

# ─── Start infrastructure services ───────────────────────────────────────────
log "Starting infrastructure services (postgres, redis, minio)..."
docker compose up -d postgres redis minio
sleep 5

# Wait for postgres
log "Waiting for PostgreSQL to be ready..."
until docker compose exec -T postgres pg_isready -U pde_user > /dev/null 2>&1; do
    sleep 2
done
log "  ✓ PostgreSQL ready"

# ─── Initialize database ──────────────────────────────────────────────────────
log "Initializing database schema..."
docker compose run --rm \
    -e POSTGRES_HOST=postgres \
    -e POSTGRES_DB="${POSTGRES_DB:-pde_simulations}" \
    -e POSTGRES_USER="${POSTGRES_USER:-pde_user}" \
    -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-pde_secret_change_me}" \
    agents python -c "
from database.operations import init_db
init_db()
print('Database tables created successfully.')
" 2>&1 | grep -v "^$" || warn "Database init failed - will retry on first use (check 'docker compose logs agents')"

# ─── Start Ollama and pull models ─────────────────────────────────────────────
log "Starting Ollama LLM server..."
docker compose up -d ollama
log "  Waiting for Ollama to initialize (up to 30s)..."
for i in $(seq 1 15); do
    if docker compose exec -T ollama ollama list > /dev/null 2>&1; then
        log "  ✓ Ollama ready"
        break
    fi
    sleep 2
done

log "Pulling LLM models (this can take 30-60 minutes on first run)..."
log "  Models: qwen2.5-coder:32b, qwen2.5-coder:14b, llama3.3:70b"
log "  You can monitor progress with: docker compose logs -f ollama"
docker compose exec -T ollama bash /ollama-init.sh || warn "Model pull failed - run 'make pull-models' to retry"

# ─── Start remaining services ─────────────────────────────────────────────────
log "Starting all services..."
docker compose up -d

sleep 10

# ─── Health check ─────────────────────────────────────────────────────────────
echo ""
log "Running health checks..."

check_service() {
    local name="$1"
    local url="$2"
    if curl -sf "$url" > /dev/null 2>&1; then
        log "  ✓ $name: $url"
    else
        warn "  ✗ $name not ready yet: $url"
    fi
}

check_service "Ollama"         "http://localhost:11434/api/tags"
check_service "Agents API"     "http://localhost:8000/health"
check_service "Dashboard"      "http://localhost:8050"
check_service "JupyterLab"     "http://localhost:8888"
check_service "MLflow"         "http://localhost:5000"
check_service "MinIO Console"  "http://localhost:9001"

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Setup Complete!                            ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Service          URL                                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Dashboard        http://localhost:8050                 ║"
echo "║  Agents API       http://localhost:8000                 ║"
echo "║  JupyterLab       http://localhost:8888                 ║"
echo "║  MLflow           http://localhost:5000                 ║"
echo "║  MinIO Console    http://localhost:9001                 ║"
echo "║  Ollama           http://localhost:11434                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Quick start:"
echo "    make run              # start everything"
echo "    make simulate-2d      # run a 2D heat equation"
echo "    make simulate-3d      # run a 3D heat equation"
echo "    make sweep            # run a parametric sweep"
echo "    make logs             # view all service logs"
echo ""
