"""
V-Code Pilot - vLLM Proxy Client
Async HTTP client for forwarding requests to vLLM backend.
"""

import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

import httpx
import orjson
import structlog

from app.config import get_settings
from app.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)

logger = structlog.get_logger(__name__)


class VLLMProxyError(Exception):
    """Exception raised for vLLM proxy errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class VLLMClient:
    """
    Async HTTP client for vLLM backend communication.
    
    Handles both streaming and non-streaming requests.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = 300.0):
        """
        Initialize the vLLM client.
        
        Args:
            base_url: vLLM server base URL. Uses config if not provided.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.vllm.api_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20
                )
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check vLLM backend health.
        
        Returns:
            Dict containing health status.
        """
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=10.0)
            
            if response.status_code == 200:
                return {"status": "healthy", "backend": "vllm"}
            else:
                return {
                    "status": "unhealthy",
                    "backend": "vllm",
                    "error": f"Status code: {response.status_code}"
                }
        except Exception as e:
            logger.error("vLLM health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "backend": "vllm",
                "error": str(e)
            }
    
    async def get_models(self) -> Dict[str, Any]:
        """
        Get available models from vLLM.
        
        Returns:
            Dict containing model list.
        """
        try:
            client = await self._get_client()
            response = await client.get("/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get models", error=str(e))
            raise VLLMProxyError(f"Failed to get models: {e}")
    
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        request_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """
        Send non-streaming chat completion request.
        
        Args:
            request: Chat completion request.
            request_id: Optional request ID for tracking.
            
        Returns:
            ChatCompletionResponse: The completion response.
        """
        request_id = request_id or str(uuid.uuid4())
        start_time = time.perf_counter()
        
        try:
            client = await self._get_client()
            
            # Prepare request payload
            payload = request.model_dump(exclude_none=True)
            payload["stream"] = False  # Ensure non-streaming
            
            logger.info(
                "Sending chat completion request",
                request_id=request_id,
                model=request.model,
                messages_count=len(request.messages)
            )
            
            response = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"X-Request-ID": request_id}
            )
            
            if response.status_code != 200:
                error_body = response.text
                logger.error(
                    "vLLM request failed",
                    request_id=request_id,
                    status_code=response.status_code,
                    error=error_body
                )
                raise VLLMProxyError(
                    f"vLLM request failed: {error_body}",
                    status_code=response.status_code
                )
            
            result = response.json()
            duration = (time.perf_counter() - start_time) * 1000
            
            logger.info(
                "Chat completion successful",
                request_id=request_id,
                duration_ms=round(duration, 2),
                prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=result.get("usage", {}).get("completion_tokens", 0)
            )
            
            return ChatCompletionResponse(**result)
            
        except httpx.TimeoutException:
            logger.error("vLLM request timeout", request_id=request_id)
            raise VLLMProxyError("Request timeout", status_code=504)
        except httpx.RequestError as e:
            logger.error("vLLM connection error", request_id=request_id, error=str(e))
            raise VLLMProxyError(f"Connection error: {e}", status_code=502)
    
    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        request_id: Optional[str] = None
    ) -> AsyncIterator[tuple[str, Optional[UsageInfo]]]:
        """
        Send streaming chat completion request.
        
        Args:
            request: Chat completion request.
            request_id: Optional request ID for tracking.
            
        Yields:
            Tuple of (SSE data string, usage info if final chunk).
        """
        request_id = request_id or str(uuid.uuid4())
        start_time = time.perf_counter()
        usage_info: Optional[UsageInfo] = None
        
        try:
            client = await self._get_client()
            
            # Prepare request payload
            payload = request.model_dump(exclude_none=True)
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            
            logger.info(
                "Starting streaming chat completion",
                request_id=request_id,
                model=request.model,
                messages_count=len(request.messages)
            )
            
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                headers={"X-Request-ID": request_id}
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(
                        "vLLM streaming request failed",
                        request_id=request_id,
                        status_code=response.status_code,
                        error=error_body.decode()
                    )
                    raise VLLMProxyError(
                        f"vLLM request failed: {error_body.decode()}",
                        status_code=response.status_code
                    )
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        
                        if data == "[DONE]":
                            yield "data: [DONE]\n\n", usage_info
                            break
                        
                        try:
                            chunk_data = orjson.loads(data)
                            
                            # Extract usage info from final chunk
                            if "usage" in chunk_data and chunk_data["usage"]:
                                usage_info = UsageInfo(**chunk_data["usage"])
                            
                            yield f"data: {data}\n\n", None
                            
                        except orjson.JSONDecodeError:
                            logger.warning(
                                "Failed to parse SSE chunk",
                                request_id=request_id,
                                data=data[:100]
                            )
                            continue
            
            duration = (time.perf_counter() - start_time) * 1000
            
            logger.info(
                "Streaming chat completion finished",
                request_id=request_id,
                duration_ms=round(duration, 2),
                prompt_tokens=usage_info.prompt_tokens if usage_info else 0,
                completion_tokens=usage_info.completion_tokens if usage_info else 0
            )
            
        except httpx.TimeoutException:
            logger.error("vLLM streaming request timeout", request_id=request_id)
            raise VLLMProxyError("Request timeout", status_code=504)
        except httpx.RequestError as e:
            logger.error(
                "vLLM streaming connection error",
                request_id=request_id,
                error=str(e)
            )
            raise VLLMProxyError(f"Connection error: {e}", status_code=502)


# Global client instance
_vllm_client: Optional[VLLMClient] = None


async def get_vllm_client() -> VLLMClient:
    """Get the global vLLM client instance."""
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = VLLMClient()
    return _vllm_client


async def close_vllm_client() -> None:
    """Close the global vLLM client."""
    global _vllm_client
    if _vllm_client:
        await _vllm_client.close()
        _vllm_client = None
