# V-Code Pilot - LLM as a Service Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.6.4-green.svg)](https://github.com/vllm-project/vllm)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready **LLM as a Service (LLaaS)** platform providing an OpenAI-compatible API for AI Copilot/Coding assistance with token-based usage management.

## 🚀 Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI Chat Completions API
- **Streaming Support**: Real-time token streaming for responsive Copilot experience
- **Token-Based Billing**: Automatic usage tracking and balance management
- **X-API-KEY Authentication**: Secure API access with simple key-based auth
- **Resource Isolation**: GPU memory limits and CPU/RAM constraints for shared environments
- **Production Ready**: Structured logging, health checks, graceful shutdown

## 🖥️ Hardware Requirements

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Tesla V100S (32GB VRAM) or equivalent |
| GPU Architecture | Volta (SM 7.0) - Uses float16/GPTQ (no AWQ/BF16) |
| System RAM | Minimum 16GB dedicated |
| CPU Cores | Minimum 4 cores dedicated |
| Storage | 100GB+ for model weights |

## 📦 Model Configuration

- **Model**: `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4`
- **Quantization**: GPTQ (4-bit) - Required for V100S
- **Data Type**: float16 - Volta architecture compatible
- **Context Length**: 8192 tokens (configurable)
- **GPU Memory Utilization**: 70% (preserves headroom for shared servers)

## 🏗️ Project Structure

```
bida-llm-service/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── auth.py               # API key authentication
│   ├── config.py             # Pydantic settings management
│   ├── logging_config.py     # Structured logging setup
│   ├── models.py             # Pydantic request/response models
│   ├── token_manager.py      # SQLite token balance management
│   └── vllm_client.py        # Async vLLM backend client
├── main.py                   # FastAPI application
├── entrypoint.sh             # Docker startup script
├── Dockerfile                # Multi-stage container build
├── docker-compose.yml        # Production deployment config
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Clone and Configure

```bash
# Clone the repository
cd bida-llm-service

# Create environment configuration
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 2. Build and Run with Docker Compose

```bash
# Build the container
docker-compose build

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 3. Verify Installation

```bash
# Health check
curl http://localhost:8080/health

# List models
curl -H "X-API-KEY: your-api-key" http://localhost:8080/v1/models
```

## 📡 API Reference

### Authentication

All API requests require the `X-API-KEY` header:

```bash
curl -H "X-API-KEY: your-api-key-here" http://localhost:8080/v1/models
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Create chat completion |
| GET | `/v1/usage/balance` | Get token balance |
| GET | `/v1/usage/history` | Get usage history |

### Chat Completion (Non-Streaming)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key" \
  -d '{
    "model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

### Chat Completion (Streaming)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key" \
  -d '{
    "model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    "messages": [
      {"role": "user", "content": "Explain async/await in Python"}
    ],
    "stream": true,
    "max_tokens": 512
  }'
```

### Check Token Balance

```bash
curl -H "X-API-KEY: your-api-key" http://localhost:8080/v1/usage/balance
```

**Response:**
```json
{
  "api_key": "your...key",
  "balance": 95000,
  "total_used": 5000,
  "created_at": "2025-01-04T10:00:00Z",
  "last_used": "2025-01-04T15:30:00Z"
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL_NAME` | Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 | HuggingFace model ID |
| `VLLM_PORT` | 8000 | vLLM server port |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.70 | GPU memory fraction |
| `VLLM_DTYPE` | float16 | Model data type |
| `VLLM_QUANTIZATION` | gptq | Quantization method |
| `VLLM_MAX_MODEL_LEN` | 8192 | Max context length |
| `PROXY_PORT` | 8080 | FastAPI proxy port |
| `DEFAULT_TOKEN_BALANCE` | 100000 | Initial user balance |
| `TOKEN_DB_PATH` | /app/data/tokens.db | SQLite database path |
| `LOG_LEVEL` | INFO | Logging level |
| `API_KEY_HEADER` | X-API-KEY | Auth header name |

### Resource Limits (docker-compose.yml)

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 16G
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Specific GPU
          capabilities: [gpu]
```

## 🔧 Development

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start vLLM server (in terminal 1)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 \
  --dtype float16 \
  --quantization gptq \
  --gpu-memory-utilization 0.70

# Start FastAPI proxy (in terminal 2)
export VLLM_BACKEND_URL=http://localhost:8000
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### API Documentation

Once running, access the interactive API docs:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## 📊 Monitoring

### Health Checks

```bash
# Gateway health
curl http://localhost:8080/health

# Kubernetes probe
curl http://localhost:8080/healthz
```

### Logs

```bash
# Docker logs
docker-compose logs -f v-code-pilot

# Log files inside container
docker exec v-code-pilot tail -f /app/logs/vllm.log
docker exec v-code-pilot tail -f /app/logs/proxy.log
```

## 🔐 Security Considerations

1. **API Keys**: Use strong, unique API keys (minimum 32 characters recommended)
2. **Network**: Deploy behind a reverse proxy (nginx/traefik) with TLS
3. **Rate Limiting**: Consider adding rate limiting for production
4. **Token Database**: Back up SQLite database regularly
5. **Container Security**: Runs as non-root user by default

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `VLLM_GPU_MEMORY_UTILIZATION` |
| Model loading slow | First run downloads weights (~20GB) |
| Connection refused | Wait for vLLM startup (check `/health`) |
| AWQ/BF16 errors | Ensure using `float16` and `gptq` for V100S |

### Debug Mode

```bash
# Enable debug logging
docker-compose run -e LOG_LEVEL=DEBUG v-code-pilot
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

**V-Code Pilot** - Empowering developers with AI-powered coding assistance.
