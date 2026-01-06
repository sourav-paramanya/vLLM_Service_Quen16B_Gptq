"""
V-Code Pilot - Application Configuration
Centralized configuration management using Pydantic Settings.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """Hugging Face Transformers model configuration."""
    
    model_config = SettingsConfigDict(env_prefix="MODEL_")
    
    name: str = Field(
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        description="HuggingFace model identifier"
    )
    revision: str = Field(
        default="main",
        description="Model revision/branch"
    )
    max_len: int = Field(
        default=8192,
        alias="MAX_MODEL_LEN",
        description="Maximum sequence length"
    )
    device: str = Field(
        default="cuda:0",
        alias="DEVICE",
        description="Device to run the model on"
    )
    cache_dir: str = Field(
        default="/app/models",
        description="Directory to cache model weights"
    )


class ServerSettings(BaseSettings):
    """FastAPI server configuration."""
    
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, description="Server port")
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


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json/text)")


class Settings(BaseSettings):
    """Main application settings aggregating all configurations."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application metadata
    app_name: str = Field(default="V-Code Pilot")
    app_version: str = Field(default="2.0.0")
    app_description: str = Field(
        default="LLM as a Service Platform (Transformers Edition)"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Sub-configurations
    model: ModelSettings = Field(default_factory=ModelSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    tokens: TokenSettings = Field(default_factory=TokenSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
