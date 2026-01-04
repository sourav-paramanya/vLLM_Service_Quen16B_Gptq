"""
V-Code Pilot - Authentication Service
API key validation and request authentication.
"""

from typing import Optional

import structlog
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.config import get_settings
from app.token_manager import get_token_manager

logger = structlog.get_logger(__name__)

# API Key header configuration
settings = get_settings()
api_key_header = APIKeyHeader(
    name=settings.security.api_key_header,
    auto_error=False
)


class AuthenticationError(HTTPException):
    """Custom authentication exception."""
    
    def __init__(self, detail: str = "Invalid or missing API key"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": detail,
                    "type": "authentication_error",
                    "code": "invalid_api_key"
                }
            }
        )


class InsufficientBalanceError(HTTPException):
    """Custom exception for insufficient token balance."""
    
    def __init__(self, balance: int, required: int = 1):
        super().__init__(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": {
                    "message": f"Insufficient token balance. Current: {balance}, Required: {required}",
                    "type": "insufficient_balance",
                    "code": "insufficient_tokens",
                    "current_balance": balance
                }
            }
        )


async def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key.
    
    For this implementation, any non-empty API key is considered valid.
    In production, you would validate against a database or external service.
    
    Args:
        api_key: The API key to validate.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    if not api_key or len(api_key) < 8:
        return False
    
    # Basic format validation
    # In production, validate against stored keys
    return True


async def get_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """
    Extract and validate API key from request.
    
    Args:
        request: The incoming request.
        api_key: API key from header.
        
    Returns:
        str: Validated API key.
        
    Raises:
        AuthenticationError: If API key is invalid or missing.
    """
    if not api_key:
        logger.warning(
            "Missing API key",
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown"
        )
        raise AuthenticationError("API key is required")
    
    if not await validate_api_key(api_key):
        logger.warning(
            "Invalid API key format",
            api_key_prefix=api_key[:4] if len(api_key) >= 4 else "****",
            path=request.url.path
        )
        raise AuthenticationError("Invalid API key format")
    
    logger.debug(
        "API key validated",
        api_key_prefix=api_key[:4],
        path=request.url.path
    )
    
    return api_key


async def verify_token_balance(
    api_key: str,
    minimum_required: int = 1
) -> int:
    """
    Verify user has sufficient token balance.
    
    Args:
        api_key: The user's API key.
        minimum_required: Minimum tokens required.
        
    Returns:
        int: Current token balance.
        
    Raises:
        InsufficientBalanceError: If balance is insufficient.
    """
    token_manager = await get_token_manager()
    balance = await token_manager.get_balance(api_key)
    
    if balance < minimum_required:
        logger.warning(
            "Insufficient token balance",
            api_key_prefix=api_key[:4],
            balance=balance,
            required=minimum_required
        )
        raise InsufficientBalanceError(balance=balance, required=minimum_required)
    
    return balance


async def authenticate_request(
    request: Request,
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """
    Full authentication flow: validate key and check balance.
    
    Args:
        request: The incoming request.
        api_key: API key from header.
        
    Returns:
        str: Validated API key.
        
    Raises:
        AuthenticationError: If authentication fails.
        InsufficientBalanceError: If balance is insufficient.
    """
    validated_key = await get_api_key(request, api_key)
    await verify_token_balance(validated_key)
    return validated_key
