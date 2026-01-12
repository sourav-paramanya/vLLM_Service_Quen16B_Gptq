## 📊 Model Performance & Capability

Our platform supports state-of-the-art LLMs (e.g., Qwen2.5-14B-Instruct-GPTQ-Int4) with the following capabilities:

- **Instruction Following**: Responds accurately to user prompts and coding tasks.
- **Streaming Responses**: Delivers tokens in real-time for interactive applications.
- **Quantized Inference**: Efficient 4-bit quantization (GPTQ) for reduced memory and faster inference.
- **Scalability**: Handles multiple concurrent requests with GPU acceleration.
- **Context Length**: Supports up to 8K tokens per request (model-dependent).
- **Performance**: On recommended hardware (16GB+ VRAM), typical response latency is under 2 seconds for short prompts; throughput scales with GPU and batch size.

**Note:** Actual performance and capability depend on the selected model, hardware, and configuration. For best results, use models optimized for instruction and chat tasks.

# Local LLM Service

A modular LLM-as-a-Service platform with OpenAI-compatible API endpoints, model hosting, and client integration. This project uses Hugging Face Transformers and vLLM for efficient inference.


## 🚀 Features

- OpenAI-compatible API endpoints
- Hugging Face Transformers and vLLM backend
- Dockerized deployment with resource isolation
- Streamlit client for interactive usage
- Nginx reverse proxy configuration


## 🖥️ Hardware Requirements

| Component      | Specification                |
|---------------|------------------------------|
| GPU           | Recommended: 16GB+ VRAM      |
| System RAM    | Minimum 16GB                 |
| CPU Cores     | Minimum 4                    |


## 📦 Model Configuration

- Place your Hugging Face models in the `hf_models/` directory.
- Example:
  ```bash
  git lfs install
  git clone https://huggingface.co/<model-repo> hf_models/<model-folder>
  ```


## 🏗️ Project Structure

```
bida-llm-service/
├── client/
│   └── sreamlitClient.py         # Streamlit client app
├── hf_models/                    # Hugging Face models (downloaded)
├── nginx/
│   └── default.conf/             # Nginx reverse proxy config
├── testing/
│   ├── environment.yml           # Test environment dependencies
│   └── testApp.py                # Test scripts
├── vllm-server/
│   ├── Dockerfile                # vLLM server container build
│   ├── entrypoint.sh             # vLLM server startup script
│   └── requirements.txt          # vLLM server dependencies
├── docker-compose.yml            # Multi-service orchestration
├── README.md                     # Project documentation
```


## 🚀 Quick Start

### 1. Build and Run

```bash
# Build and start all services
docker-compose up -d --build

# Check logs
docker-compose logs -f
```

### 2. Verify Installation

```bash
# Health check (if exposed)
curl http://localhost:8080/health
```


## 📡 API Usage

### Chat Completion (Streaming)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key" \
  -d '{
    "model": "<your-model>",
    "messages": [
      {"role": "user", "content": "Write a Python function to sort a list."}
    ],
    "stream": true
  }'
```


## ⚙️ Configuration

Configuration is managed via environment variables and Docker Compose. See `vllm-server/requirements.txt` and `testing/environment.yml` for dependencies.


## 📄 License

MIT License
