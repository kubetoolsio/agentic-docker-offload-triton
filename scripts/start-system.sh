#!/bin/bash
set -euo pipefail

echo "Starting AI Offload system..."

if ! command -v docker &>/dev/null; then
  echo "❌ Docker is not installed. Please install Docker and try again."
  exit 1
fi
if ! docker compose version &>/dev/null; then
  echo "❌ The 'docker compose' plugin is required."
  exit 1
fi

# Capture user‑provided overrides before sourcing .env
USER_OFFLOAD_MODE="${OFFLOAD_MODE:-}"
USER_OFFLOAD_ENABLED="${OFFLOAD_ENABLED:-}"

# Load env files (allow .env.gpu if present)
[ -f .env ] && source .env || true
[ -f .env.gpu ] && source .env.gpu || true

# Restore user overrides (only if user specified them)
[ -n "$USER_OFFLOAD_MODE" ] && OFFLOAD_MODE="$USER_OFFLOAD_MODE"
[ -n "$USER_OFFLOAD_ENABLED" ] && OFFLOAD_ENABLED="$USER_OFFLOAD_ENABLED"

# Defaults
OFFLOAD_MODE=${OFFLOAD_MODE:-auto}
OFFLOAD_ENABLED=${OFFLOAD_ENABLED:-false}
REMOTE_DOCKER_MODEL_RUNNER_URL=${REMOTE_DOCKER_MODEL_RUNNER_URL:-}

# --- Detect Local GPU (skip if forcing remote-offload) ---
has_local_gpu="false"
if [ "$OFFLOAD_MODE" != "remote-offload" ]; then
  if docker info 2>/dev/null | grep -iq nvidia; then
    has_local_gpu="true"
  else
    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
      has_local_gpu="true"
    fi
  fi
fi
echo "Detected local GPU (pre-mode resolution): $has_local_gpu"

# --- Determine Final OFFLOAD_MODE (only if auto) ---
if [ "$OFFLOAD_MODE" = "auto" ]; then
  if [ "$has_local_gpu" = "true" ]; then
    OFFLOAD_MODE="local-gpu"
  elif [ -n "$REMOTE_DOCKER_MODEL_RUNNER_URL" ]; then
    OFFLOAD_MODE="remote-offload"
  else
    OFFLOAD_MODE="cpu"
  fi
fi

case "$OFFLOAD_MODE" in
  local-gpu)
    if [ "$has_local_gpu" != "true" ]; then
      echo "⚠️  OFFLOAD_MODE=local-gpu but no local GPU found; switching to cpu."
      OFFLOAD_MODE="cpu"
      OFFLOAD_ENABLED=false
    else
      OFFLOAD_ENABLED=true
    fi
    ;;
  remote-offload)
    # Force has_local_gpu false (clarity) even if remote context lets GPU containers run
    has_local_gpu="false"
    if ! docker offload status >/dev/null 2>&1 || ! docker offload status | grep -q "Started"; then
      echo "🌩️  Starting Docker Offload GPU session..."
      docker offload start --gpu
    fi
    OFFLOAD_ENABLED=true
    ;;
  cpu)
    OFFLOAD_ENABLED=false
    ;;
  *)
    echo "❌ Unknown OFFLOAD_MODE='$OFFLOAD_MODE' (auto|local-gpu|remote-offload|cpu)."
    exit 1
    ;;
esac

echo "✅ Final OFFLOAD_MODE: $OFFLOAD_MODE"
echo "   OFFLOAD_ENABLED: $OFFLOAD_ENABLED"

export OFFLOAD_MODE OFFLOAD_ENABLED REMOTE_DOCKER_MODEL_RUNNER_URL

echo "🛑 Stopping any existing stack to ensure a clean start..."
docker compose down -v --remove-orphans >/dev/null 2>&1 || true

compose_files="-f docker-compose.yml"
if [ "$OFFLOAD_MODE" = "local-gpu" ] || [ "$OFFLOAD_MODE" = "remote-offload" ]; then
  echo "🚀 Applying GPU configuration override..."
  compose_files="$compose_files -f docker-compose.gpu.override.yml"
fi

echo "🚀 Bringing stack up in '$OFFLOAD_MODE' mode..."
docker compose $compose_files up -d

echo "⏳ Waiting for coordinator (port 8090) to become healthy..."
for i in {1..40}; do
  if curl -sf http://localhost:8090/health >/dev/null 2>&1; then
    echo "✅ Coordinator is ready!"
    break
  fi
  if [ "$i" -eq 40 ]; then
    echo "❌ Coordinator failed to become ready after 80 seconds."
    docker compose logs --tail=120 coordinator-agent || true
    exit 1
  fi
  sleep 2
done

echo ""
echo "🎉 System is up and running!"
echo ""
echo "   Mode Summary:"
echo "     OFFLOAD_MODE: $OFFLOAD_MODE"
echo "     Local GPU Detected: $has_local_gpu"
echo "     Remote Runner URL: ${REMOTE_DOCKER_MODEL_RUNNER_URL:-<none>}"
echo "     Docker Offload Active: $(docker context show 2>/dev/null | grep -q 'docker-cloud' && echo yes || echo no)"
echo ""
echo "   Next Steps:"
echo "     Run tests: ./scripts/test-system.sh"
echo "     Verify remote GPU: ./scripts/verify-remote-gpu.sh (remote or local GPU modes)"
echo ""