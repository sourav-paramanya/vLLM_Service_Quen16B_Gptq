"""
V-Code Pilot - Main FastAPI Application
Production-ready LLM as a Service API Gateway.

This module provides:
- OpenAI-compatible API endpoints
- X-API-KEY authentication
- Token-based usage tracking
- Streaming response support
- Request/response logging
"""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, AsyncIterator

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    TokenBalance,
    UsageInfo,
)
from app.token_manager import get_token_manager
from app.vllm_client import VLLMProxyError, close_vllm_client, get_vllm_client

# Configure logging on module load
configure_logging()

logger = structlog.get_logger(__name__)
settings = get_settings()


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "Starting V-Code Pilot",
        version=settings.app_version,
        vllm_backend=settings.vllm.api_url
    )
    
    # Initialize token manager
    token_manager = await get_token_manager()
    logger.info("Token manager initialized", db_path=settings.tokens.db_path)
    
    # Verify vLLM backend connectivity
    try:
        vllm_client = await get_vllm_client()
        health = await vllm_client.health_check()
        logger.info("vLLM backend status", **health)
    except Exception as e:
        logger.warning("vLLM backend not ready", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("Shutting down V-Code Pilot")
    await close_vllm_client()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(AuthenticationError)
async def auth_exception_handler(
    request: Request,
    exc: AuthenticationError
) -> JSONResponse:
    """Handle authentication errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )


@app.exception_handler(InsufficientBalanceError)
async def balance_exception_handler(
    request: Request,
    exc: InsufficientBalanceError
) -> JSONResponse:
    """Handle insufficient balance errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )


@app.exception_handler(VLLMProxyError)
async def vllm_exception_handler(
    request: Request,
    exc: VLLMProxyError
) -> JSONResponse:
    """Handle vLLM proxy errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": "backend_error",
                "code": "vllm_error"
            }
        }
    )


# =============================================================================
# Middleware
# =============================================================================

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all HTTP requests with timing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()
    
    # Add request ID to response headers
    response = await call_next(request)
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    # Log request (skip health checks to reduce noise)
    if request.url.path not in ["/health", "/healthz"]:
        log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id
        )
    
    response.headers["X-Request-ID"] = request_id
    return response


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check() -> HealthResponse:
    """
    Check the health of the API gateway and vLLM backend.
    
    Returns:
        HealthResponse: Current health status.
    """
    try:
        vllm_client = await get_vllm_client()
        vllm_health = await vllm_client.health_check()
        vllm_status = vllm_health.get("status", "unknown")
    except Exception as e:
        vllm_status = f"error: {str(e)}"
    
    overall_status = "healthy" if vllm_status == "healthy" else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        vllm_status=vllm_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        details={
            "model": settings.vllm.model_name,
            "backend_url": settings.vllm.api_url
        }
    )


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    """Kubernetes-style health probe."""
    return {"status": "ok"}


# =============================================================================
# Models Endpoint
# =============================================================================

@app.get(
    "/v1/models",
    response_model=ModelsResponse,
    tags=["Models"],
    summary="List available models"
)
async def list_models(
    api_key: Annotated[str, Depends(get_api_key)]
) -> ModelsResponse:
    """
    List available models.
    
    Returns the models available through the vLLM backend.
    """
    try:
        vllm_client = await get_vllm_client()
        models_data = await vllm_client.get_models()
        return ModelsResponse(**models_data)
    except VLLMProxyError:
        raise
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": {"message": "Failed to retrieve models", "type": "backend_error"}}
        )


# =============================================================================
# Chat Completions Endpoint
# =============================================================================

@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"},
        402: {"model": ErrorResponse, "description": "Insufficient balance"},
        502: {"model": ErrorResponse, "description": "Backend error"}
    },
    tags=["Chat"],
    summary="Create a chat completion"
)
async def create_chat_completion(
    request: Request,
    body: ChatCompletionRequest,
    api_key: Annotated[str, Depends(authenticate_request)]
):
    """
    Create a chat completion.
    
    This endpoint is fully compatible with the OpenAI Chat Completions API.
    Supports both streaming and non-streaming responses.
    
    Token usage is automatically tracked and deducted from the user's balance.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    logger.info(
        "Chat completion request",
        request_id=request_id,
        api_key_prefix=api_key[:4],
        model=body.model,
        stream=body.stream,
        messages_count=len(body.messages)
    )
    
    vllm_client = await get_vllm_client()
    token_manager = await get_token_manager()
    
    if body.stream:
        # Streaming response
        return StreamingResponse(
            stream_chat_completion(
                vllm_client=vllm_client,
                token_manager=token_manager,
                request=body,
                api_key=api_key,
                request_id=request_id
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id
            }
        )
    else:
        # Non-streaming response
        response = await vllm_client.chat_completion(body, request_id)
        
        # Deduct tokens
        if response.usage:
            await token_manager.deduct_tokens(
                api_key=api_key,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                model=body.model,
                request_id=request_id
            )
        
        return response


async def stream_chat_completion(
    vllm_client,
    token_manager,
    request: ChatCompletionRequest,
    api_key: str,
    request_id: str
) -> AsyncIterator[str]:
    """
    Stream chat completion chunks and handle token deduction.
    
    Args:
        vllm_client: The vLLM client instance.
        token_manager: The token manager instance.
        request: Chat completion request.
        api_key: User's API key.
        request_id: Request identifier.
        
    Yields:
        SSE-formatted data chunks.
    """
    usage_info: UsageInfo | None = None
    
    try:
        async for chunk, final_usage in vllm_client.chat_completion_stream(
            request,
            request_id
        ):
            if final_usage:
                usage_info = final_usage
            yield chunk
        
        # Deduct tokens after streaming completes
        if usage_info:
            await token_manager.deduct_tokens(
                api_key=api_key,
                prompt_tokens=usage_info.prompt_tokens,
                completion_tokens=usage_info.completion_tokens,
                model=request.model,
                request_id=request_id
            )
            
    except VLLMProxyError as e:
        error_chunk = {
            "error": {
                "message": e.message,
                "type": "backend_error",
                "code": "stream_error"
            }
        }
        yield f"data: {str(error_chunk)}\n\n"
    except Exception as e:
        logger.error(
            "Streaming error",
            request_id=request_id,
            error=str(e)
        )
        error_chunk = {
            "error": {
                "message": "Internal streaming error",
                "type": "internal_error"
            }
        }
        yield f"data: {str(error_chunk)}\n\n"


# =============================================================================
# Token Management Endpoints
# =============================================================================

@app.get(
    "/v1/usage/balance",
    response_model=TokenBalance,
    tags=["Usage"],
    summary="Get token balance"
)
async def get_balance(
    api_key: Annotated[str, Depends(get_api_key)]
) -> TokenBalance:
    """
    Get the current token balance for the authenticated user.
    
    Returns:
        TokenBalance: User's token balance and usage information.
    """
    token_manager = await get_token_manager()
    return await token_manager.get_or_create_user(api_key)


@app.get(
    "/v1/usage/history",
    tags=["Usage"],
    summary="Get usage history"
)
async def get_usage_history(
    api_key: Annotated[str, Depends(get_api_key)],
    limit: int = 100
):
    """
    Get token usage history for the authenticated user.
    
    Args:
        limit: Maximum number of records to return.
        
    Returns:
        List of usage log entries.
    """
    token_manager = await get_token_manager()
    history = await token_manager.get_usage_history(api_key, limit)
    return {"usage_history": history}


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "docs": "/docs",
        "health": "/health"
    }


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.proxy.host,
        port=settings.proxy.port,
        workers=settings.proxy.workers,
        reload=settings.proxy.reload,
        log_level=settings.logging.level.lower()
    )
