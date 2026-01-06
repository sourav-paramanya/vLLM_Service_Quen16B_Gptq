"""
V-Code Pilot - Pydantic Models
Request/Response schemas for OpenAI-compatible API.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# Enumerations
# =============================================================================

class Role(str, Enum):
    """Message roles for chat completion."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Completion finish reasons."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"


# =============================================================================
# Chat Completion Models
# =============================================================================

class ChatMessage(BaseModel):
    """Individual chat message."""
    role: Role
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        description="Model identifier"
    )
    messages: List[ChatMessage] = Field(
        ...,
        description="List of messages in the conversation"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability"
    )
    top_k: int = Field(
        default=-1,
        description="Top-k sampling"
    )
    n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of completions to generate"
    )
    max_tokens: Optional[int] = Field(
        default=2048,
        ge=1,
        description="Maximum tokens to generate"
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming responses"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Stop sequences"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None,
        description="Token logit biases"
    )
    user: Optional[str] = Field(
        default=None,
        description="User identifier for tracking"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Available tools for the model"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Tool selection mode"
    )


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(description="Tokens in the prompt")
    completion_tokens: int = Field(description="Tokens in the completion")
    total_tokens: int = Field(description="Total tokens used")


class ChatCompletionChoice(BaseModel):
    """Individual completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[FinishReason] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(description="Unique completion ID")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(description="Unix timestamp")
    model: str = Field(description="Model used")
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


# =============================================================================
# Streaming Models
# =============================================================================

class DeltaMessage(BaseModel):
    """Delta content for streaming."""
    role: Optional[Role] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class StreamChoice(BaseModel):
    """Streaming choice delta."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[FinishReason] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


# =============================================================================
# Token Management Models
# =============================================================================

class TokenBalance(BaseModel):
    """User token balance information."""
    api_key: str = Field(description="User API key (masked)")
    balance: int = Field(description="Current token balance")
    total_used: int = Field(default=0, description="Total tokens used")
    created_at: str = Field(description="Account creation timestamp")
    last_used: Optional[str] = Field(
        default=None,
        description="Last usage timestamp"
    )


class TokenUsageLog(BaseModel):
    """Token usage log entry."""
    api_key: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    timestamp: str
    request_id: str


class TokenDeductionResult(BaseModel):
    """Result of token deduction operation."""
    success: bool
    previous_balance: int
    current_balance: int
    tokens_deducted: int
    message: str


# =============================================================================
# API Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    vllm_status: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Any] = Field(
        description="Error details",
        examples=[{
            "message": "Invalid API key",
            "type": "authentication_error",
            "code": "invalid_api_key"
        }]
    )


class ModelInfo(BaseModel):
    """Model information response."""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "v-code-pilot"
    permission: List[Dict[str, Any]] = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    """List models response."""
    object: Literal["list"] = "list"
    data: List[ModelInfo]
