# V-Code Pilot - LLM as a Service Platform
# Optimized Dockerfile for NVIDIA Tesla V100S (Volta Architecture)
# Base: Ubuntu 22.04 + CUDA 12.4

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL maintainer="V-Code Pilot Team"
LABEL description="Production LLaaS Platform for AI Copilot Assistance"
LABEL version="1.0.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Python and CUDA environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# CUDA configuration for Tesla V100S (Volta - Compute Capability 7.0)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="7.0"

# vLLM specific optimizations for V100S
ENV VLLM_ATTENTION_BACKEND=FLASHINFER
ENV VLLM_USE_TRITON_FLASH_ATTN=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    libssl-dev \
    libffi-dev \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Create application directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r vcodepilot && useradd -r -g vcodepilot vcodepilot

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models \
    && chown -R vcodepilot:vcodepilot /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: vLLM installation for V100S requires specific torch version
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install -r requirements.txt

# Copy application code
COPY --chown=vcodepilot:vcodepilot . .

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose ports
# 8000: vLLM OpenAI-compatible API
# 8080: FastAPI Proxy Gateway
EXPOSE 8000 8080

# Health check for the FastAPI proxy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER vcodepilot

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
