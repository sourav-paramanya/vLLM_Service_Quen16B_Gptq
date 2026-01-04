"""
V-Code Pilot - Application Configuration
Centralized configuration management using Pydantic Settings.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLLMSettings(BaseSettings):
    """vLLM inference engine configuration."""
    
    model_config = SettingsConfigDict(env_prefix="VLLM_")
    
    model_name: str = Field(
        default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        description="HuggingFace model identifier"
    )
    host: str = Field(default="0.0.0.0", description="vLLM server host")
    port: int = Field(default=8000, description="vLLM server port")
    gpu_memory_utilization: float = Field(
        default=0.70,
        ge=0.1,
        le=0.95,
        description="GPU memory utilization fraction"
    )
    dtype: str = Field(
        default="float16",
        description="Model data type (float16 for V100S)"
    )
    quantization: str = Field(
        default="gptq",
        description="Quantization method (gptq for V100S)"
    )
    max_model_len: int = Field(
        default=8192,
        description="Maximum sequence length"
    )
    tensor_parallel_size: int = Field(
        default=1,
        description="Number of GPUs for tensor parallelism"
    )
    backend_url: Optional[str] = Field(
        default=None,
        description="Override backend URL (auto-constructed if not set)"
    )
    
    @property
    def api_url(self) -> str:
        """Get the vLLM API base URL."""
        if self.backend_url:
            return self.backend_url
        return f"http://{self.host}:{self.port}"


class ProxySettings(BaseSettings):
    """FastAPI proxy configuration."""
    
    model_config = SettingsConfigDict(env_prefix="PROXY_")
    
    host: str = Field(default="0.0.0.0", description="Proxy server host")
    port: int = Field(default=8080, description="Proxy server port")
    workers: int = Field(default=1, description="Number of uvicorn workers")
    reload: bool = Field(default=False, description="Enable auto-reload")


class TokenSettings(BaseSettings):
    """Token management configuration."""
    
    model_config = SettingsConfigDict(env_prefix="TOKEN_")
    
    db_path: str = Field(
        default="/app/data/tokens.db",
        alias="TOKEN_DB_PATH",
        description="SQLite database path for token storage"
    )
    default_balance: int = Field(
        default=100000,
        alias="DEFAULT_TOKEN_BALANCE",
        description="Default token balance for new users"
    )


class SecuritySettings(BaseSettings):
    """Security configuration."""
    
    api_key_header: str = Field(
        default="X-API-KEY",
        description="Header name for API key authentication"
    )
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Maximum requests per minute"
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json/text)")
    file_path: Optional[str] = Field(
        default="/app/logs/v-code-pilot.log",
        description="Log file path"
    )


class Settings(BaseSettings):
    """Main application settings aggregating all configurations."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application metadata
    app_name: str = Field(default="V-Code Pilot")
    app_version: str = Field(default="1.0.0")
    app_description: str = Field(
        default="LLM as a Service Platform for AI Copilot Assistance"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Sub-configurations
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    proxy: ProxySettings = Field(default_factory=ProxySettings)
    tokens: TokenSettings = Field(default_factory=TokenSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings: Application configuration singleton.
    """
    return Settings()
