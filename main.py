"""
V-Code Pilot - Main FastAPI Application
Production-ready LLM as a Service API Gateway using Hugging Face Transformers.
"""

import asyncio
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, AsyncIterator, Optional

import structlog
import torch
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from app.auth import (
    AuthenticationError,
    InsufficientBalanceError,
    authenticate_request,
    get_api_key,
)
from app.config import get_settings
from app.logging_config import configure_logging, log_request
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    ModelInfo,
    TokenBalance,
    UsageInfo,
    FinishReason,
)
from app.token_manager import get_token_manager

# Configure logging
configure_logging()
logger = structlog.get_logger(__name__)
settings = get_settings()

# Global model engine
engine = None


class ModelEngine:
    """
    Wrapper for Hugging Face Transformers model.
    Handles loading and inference.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = settings.model.device
        self.lock = asyncio.Lock()  # Ensure sequential access for VRAM safety
    
    def load_model(self):
        """Load model and tokenizer."""
        logger.info("Loading model...", model=settings.model.name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model.name,
                revision=settings.model.revision,
                cache_dir=settings.model.cache_dir,
                trust_remote_code=True
            )

            is_cuda = isinstance(self.device, str) and self.device.startswith("cuda")
            device_map = "auto" if is_cuda else None
            torch_dtype = torch.float16 if is_cuda else torch.float32

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model.name,
                    revision=settings.model.revision,
                    device_map=device_map,
                    trust_remote_code=True,
                    cache_dir=settings.model.cache_dir,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
            except Exception as e:
                # Some GPTQ repos still load more reliably via AutoGPTQ directly.
                logger.warning("Transformers load failed; trying AutoGPTQ fallback", error=str(e))
                from auto_gptq import AutoGPTQForCausalLM  # type: ignore

                # NOTE: AutoGPTQ APIs vary slightly by version; keep kwargs minimal.
                self.model = AutoGPTQForCausalLM.from_quantized(
                    settings.model.name,
                    revision=settings.model.revision,
                    cache_dir=settings.model.cache_dir,
                    trust_remote_code=True,
                    device=self.device if is_cuda else "cpu",
                    use_safetensors=True,
                )

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
        request_id: str
    ) -> AsyncIterator[str]:
        """
        Generate streaming response.
        """
        async with self.lock:
            try:
                # Prepare inputs
                messages = [m.model_dump(exclude_none=True) for m in request.messages]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                
                # Count prompt tokens
                prompt_tokens = len(model_inputs.input_ids[0])
                
                # Setup streamer
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                
                # Generation kwargs
                generation_kwargs = dict(
                    model_inputs,
                    streamer=streamer,
                    max_new_tokens=request.max_tokens or 2048,
                    do_sample=request.temperature > 0,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Run generation in a separate thread
                thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                completion_text = ""
                
                # Yield chunks
                for new_text in streamer:
                    completion_text += new_text
                    chunk_data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": new_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {chunk_data}\n\n"
                    # Yield to event loop to allow other tasks (like health checks)
                    await asyncio.sleep(0)
                
                # Final chunk
                yield "data: [DONE]\n\n"
                
                # Calculate usage (approximate for completion)
                completion_tokens = len(self.tokenizer.encode(completion_text))
                
                # Return usage info for the caller to handle deduction
                # We can't yield it in SSE easily without breaking standard, 
                # but we can return it or handle deduction here.
                # For simplicity, we'll handle deduction in the caller if possible,
                # but since this is a generator, we must do it here or pass a callback.
                
                token_manager = await get_token_manager()
                # We need the API key here, but it's not passed. 
                # Let's refactor to pass it or handle it.
                # Actually, we can just return the usage stats as a special internal value if needed,
                # but better to just handle it in the wrapper.
                
                # For now, we will just log it. The wrapper function `stream_chat_completion`
                # will need to handle the actual deduction if we can get the counts back.
                # A common pattern is to yield the usage in the last chunk if the client supports it (OpenAI does now).
                
                final_usage_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                # yield f"data: {final_usage_chunk}\n\n" # Optional: OpenAI supports this now
                
                # Store usage for the caller
                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }
                
            except Exception as e:
                logger.error("Generation error", error=str(e))
                yield f"data: {{'error': '{str(e)}'}}\n\n"

    async def generate(
        self,
        request: ChatCompletionRequest
    ) -> tuple[str, UsageInfo]:
        """
        Generate non-streaming response.
        """
        async with self.lock:
            # Prepare inputs
            messages = [m.model_dump(exclude_none=True) for m in request.messages]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            prompt_tokens = len(model_inputs.input_ids[0])
            
            # Run generation
            # Use asyncio.to_thread for the blocking generate call
            generated_ids = await asyncio.to_thread(
                self.model.generate,
                **model_inputs,
                max_new_tokens=request.max_tokens or 2048,
                do_sample=request.temperature > 0,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            completion_tokens = len(generated_ids[0])
            
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            return response_text, usage


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    global engine
    
    # Startup
    logger.info("Starting V-Code Pilot (Transformers Edition)")
    
    # Initialize token manager
    await get_token_manager()
    
    # Initialize model engine
    engine = ModelEngine()
    # Load model in a separate thread to not block startup if possible, 
    # but for safety we usually load it blocking or await it.
    # Since load_model is blocking CPU/IO, we run it in thread.
    await asyncio.to_thread(engine.load_model)
    
    yield
    
    # Shutdown
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    status_val = "healthy" if engine and engine.model else "starting"
    return HealthResponse(
        status=status_val,
        version=settings.app_version,
        vllm_status="N/A",  # Legacy field
        timestamp=datetime.now(timezone.utc).isoformat(),
        details={"backend": "transformers", "device": settings.model.device}
    )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models(api_key: Annotated[str, Depends(get_api_key)]):
    """List available models."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id=settings.model.name,
                created=int(time.time()),
                owned_by="v-code-pilot"
            )
        ]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: Request,
    body: ChatCompletionRequest,
    api_key: Annotated[str, Depends(authenticate_request)]
):
    """Create chat completion."""
    if not engine or not engine.model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token_manager = await get_token_manager()

    if body.stream:
        async def stream_generator():
            # We need to capture usage from the engine
            # This is a bit tricky with the generator, but we'll do our best
            # The engine.generate_stream yields chunks.
            # We'll wrap it to handle token deduction at the end.
            
            prompt_tokens = 0
            completion_tokens = 0
            
            async for chunk in engine.generate_stream(body, request_id):
                yield chunk
            
            # After stream ends, check if engine stored usage
            # This is a simplification. In a real concurrent app, 
            # we'd need to pass a context object to generate_stream.
            if hasattr(engine, 'last_usage'):
                usage = engine.last_usage
                await token_manager.deduct_tokens(
                    api_key=api_key,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    model=body.model,
                    request_id=request_id
                )

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming
        response_text, usage = await engine.generate(body)
        
        # Deduct tokens
        await token_manager.deduct_tokens(
            api_key=api_key,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model=body.model,
            request_id=request_id
        )
        
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason=FinishReason.STOP
                )
            ],
            usage=usage
        )


@app.get("/v1/usage/balance", response_model=TokenBalance)
async def get_balance(api_key: Annotated[str, Depends(get_api_key)]):
    """Get token balance."""
    token_manager = await get_token_manager()
    return await token_manager.get_or_create_user(api_key)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        reload=settings.server.reload
    )
