"""
V-Code Pilot - Structured Logging Configuration
JSON-formatted logging with structlog.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.types import Processor

from app.config import get_settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Determine log level
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    
    # Common processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    if settings.logging.format == "json":
        # JSON format for production
        processors: list[Processor] = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_request_logger(request_id: str, **kwargs: Any) -> structlog.BoundLogger:
    """
    Get a logger bound with request context.
    
    Args:
        request_id: Unique request identifier.
        **kwargs: Additional context to bind.
        
    Returns:
        structlog.BoundLogger: Logger with bound context.
    """
    return structlog.get_logger().bind(
        request_id=request_id,
        **kwargs
    )


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    **extra: Any
) -> None:
    """
    Log an HTTP request.
    
    Args:
        method: HTTP method.
        path: Request path.
        status_code: Response status code.
        duration_ms: Request duration in milliseconds.
        **extra: Additional fields to log.
    """
    logger = structlog.get_logger()
    
    log_data: Dict[str, Any] = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        **extra
    }
    
    if status_code >= 500:
        logger.error("Request failed", **log_data)
    elif status_code >= 400:
        logger.warning("Request error", **log_data)
    else:
        logger.info("Request completed", **log_data)
