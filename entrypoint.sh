#!/bin/bash
# V-Code Pilot - Entrypoint Script
# Transformers Edition

set -e

# =============================================================================
# Configuration
# =============================================================================

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# =============================================================================
# Start Application
# =============================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting V-Code Pilot (Transformers Edition)..."
echo "  Model: ${MODEL_NAME}"
echo "  Device: ${MODEL_DEVICE:-${DEVICE}}"

# Ensure directories exist
mkdir -p /app/data /app/logs /app/models

# Start FastAPI
exec python -m uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --no-access-log
