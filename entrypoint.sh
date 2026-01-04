#!/bin/bash
# V-Code Pilot - Entrypoint Script
# Safely starts both vLLM inference server and FastAPI proxy gateway.

set -e

# =============================================================================
# Configuration from environment variables
# =============================================================================

# vLLM Configuration
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-gptq}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

# FastAPI Proxy Configuration
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
PROXY_PORT="${PROXY_PORT:-8080}"

# Log Configuration
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1"
}

# Cleanup function for graceful shutdown
cleanup() {
    log_info "Received shutdown signal, stopping services..."
    
    # Kill vLLM server
    if [ -n "$VLLM_PID" ]; then
        log_info "Stopping vLLM server (PID: $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    
    # Kill FastAPI proxy
    if [ -n "$PROXY_PID" ]; then
        log_info "Stopping FastAPI proxy (PID: $PROXY_PID)..."
        kill -TERM "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
    fi
    
    log_info "Shutdown complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Wait for vLLM to be ready
wait_for_vllm() {
    local max_attempts=120  # 10 minutes (model loading can take time)
    local attempt=1
    
    log_info "Waiting for vLLM server to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            log_info "vLLM server is ready!"
            return 0
        fi
        
        if [ $((attempt % 10)) -eq 0 ]; then
            log_info "Still waiting for vLLM... (attempt $attempt/$max_attempts)"
        fi
        
        sleep 5
        attempt=$((attempt + 1))
    done
    
    log_error "vLLM server failed to start within timeout"
    return 1
}

# =============================================================================
# Environment Validation
# =============================================================================

log_info "=========================================="
log_info "V-Code Pilot - LLM as a Service Platform"
log_info "=========================================="

# Check NVIDIA GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    log_warn "nvidia-smi not found. GPU may not be available."
else
    log_info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi

# Display configuration
log_info "Configuration:"
log_info "  Model: ${VLLM_MODEL_NAME}"
log_info "  vLLM Port: ${VLLM_PORT}"
log_info "  Proxy Port: ${PROXY_PORT}"
log_info "  GPU Memory Utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
log_info "  Data Type: ${VLLM_DTYPE}"
log_info "  Quantization: ${VLLM_QUANTIZATION}"
log_info "  Max Model Length: ${VLLM_MAX_MODEL_LEN}"

# =============================================================================
# Start vLLM Server
# =============================================================================

log_info "Starting vLLM inference server..."

python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --dtype "${VLLM_DTYPE}" \
    --quantization "${VLLM_QUANTIZATION}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --trust-remote-code \
    --disable-log-requests \
    2>&1 | tee -a /app/logs/vllm.log &

VLLM_PID=$!
log_info "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM to be ready before starting the proxy
if ! wait_for_vllm; then
    log_error "Failed to start vLLM server"
    cleanup
    exit 1
fi

# =============================================================================
# Start FastAPI Proxy
# =============================================================================

log_info "Starting FastAPI proxy gateway..."

# Ensure data directory exists
mkdir -p /app/data /app/logs

# Set the backend URL for the proxy
export VLLM_BACKEND_URL="http://localhost:${VLLM_PORT}"

python -m uvicorn main:app \
    --host "${PROXY_HOST}" \
    --port "${PROXY_PORT}" \
    --log-level "${LOG_LEVEL,,}" \
    --no-access-log \
    2>&1 | tee -a /app/logs/proxy.log &

PROXY_PID=$!
log_info "FastAPI proxy started with PID: $PROXY_PID"

# =============================================================================
# Health Monitoring Loop
# =============================================================================

log_info "=========================================="
log_info "V-Code Pilot is now running!"
log_info "  vLLM API: http://localhost:${VLLM_PORT}"
log_info "  Proxy API: http://localhost:${PROXY_PORT}"
log_info "  Documentation: http://localhost:${PROXY_PORT}/docs"
log_info "=========================================="

# Monitor both processes
while true; do
    # Check if vLLM is still running
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log_error "vLLM server (PID: $VLLM_PID) has stopped unexpectedly!"
        cleanup
        exit 1
    fi
    
    # Check if proxy is still running
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
        log_error "FastAPI proxy (PID: $PROXY_PID) has stopped unexpectedly!"
        cleanup
        exit 1
    fi
    
    sleep 30
done
