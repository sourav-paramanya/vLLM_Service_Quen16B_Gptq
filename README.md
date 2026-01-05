# V-Code Pilot - LLM as a Service Platform (Transformers Edition)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/Transformers-4.46-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready **LLM as a Service (LLaaS)** platform providing an OpenAI-compatible API for AI Copilot/Coding assistance. This version uses **Hugging Face Transformers** and **AutoGPTQ** for inference.

## 🚀 Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI Chat Completions API
- **Hugging Face Transformers**: Uses `AutoModelForCausalLM` with GPTQ quantization
- **Streaming Support**: Real-time token streaming using `TextIteratorStreamer`
- **Token-Based Billing**: Automatic usage tracking and balance management
- **X-API-KEY Authentication**: Secure API access
- **Resource Isolation**: Docker-based GPU/CPU/RAM limits

## 🖥️ Hardware Requirements

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Tesla V100S (32GB VRAM) |
| GPU Architecture | Volta (SM 7.0) |
| System RAM | Minimum 16GB dedicated |
| CPU Cores | Minimum 4 cores dedicated |

## 📦 Model Configuration

- **Model**: `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4`
- **Quantization**: GPTQ (4-bit)
- **Backend**: Hugging Face Transformers + AutoGPTQ
- **Device**: CUDA:0

## 🏗️ Project Structure

```
bida-llm-service/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── auth.py               # API key authentication
│   ├── config.py             # Configuration settings
│   ├── logging_config.py     # Structured logging
│   ├── models.py             # Pydantic schemas
│   └── token_manager.py      # Token management
├── main.py                   # FastAPI application & Inference Engine
├── entrypoint.sh             # Startup script
├── Dockerfile                # Container build
├── docker-compose.yml        # Deployment config
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## 🚀 Quick Start

### 1. Build and Run

```bash
# Build the container
docker-compose build

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 2. Verify Installation

```bash
# Health check
curl http://localhost:8080/health
```

## 📡 API Usage

### Chat Completion (Streaming)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key" \
  -d '{
    "model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    "messages": [
      {"role": "user", "content": "Write a Python function to sort a list."}
    ],
    "stream": true
  }'
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 | Model ID |
| `DEVICE` | cuda:0 | GPU device |
| `MAX_MODEL_LEN` | 8192 | Max sequence length |
| `DEFAULT_TOKEN_BALANCE` | 100000 | Initial user balance |

## 📄 License

MIT License
