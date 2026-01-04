"""
V-Code Pilot - Token Management Service
SQLite-based token balance tracking and usage logging.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional

import aiosqlite
import structlog

from app.config import get_settings
from app.models import TokenBalance, TokenDeductionResult, TokenUsageLog

logger = structlog.get_logger(__name__)


class TokenManager:
    """
    Async token management service using SQLite.
    
    Handles user token balances, usage tracking, and deductions.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the token manager.
        
        Args:
            db_path: Path to SQLite database. Uses config default if not provided.
        """
        settings = get_settings()
        self.db_path = db_path or settings.tokens.db_path
        self.default_balance = settings.tokens.default_balance
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Create users table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        api_key TEXT PRIMARY KEY,
                        balance INTEGER NOT NULL DEFAULT 100000,
                        total_used INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        last_used TEXT
                    )
                """)
                
                # Create usage logs table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS usage_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        api_key TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL,
                        completion_tokens INTEGER NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (api_key) REFERENCES users(api_key)
                    )
                """)
                
                # Create indexes for performance
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_usage_api_key 
                    ON usage_logs(api_key)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_usage_timestamp 
                    ON usage_logs(timestamp)
                """)
                
                await db.commit()
            
            self._initialized = True
            logger.info("Token database initialized", db_path=self.db_path)
    
    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a database connection with proper initialization."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db
    
    async def get_or_create_user(self, api_key: str) -> TokenBalance:
        """
        Get user by API key, creating if not exists.
        
        Args:
            api_key: The user's API key.
            
        Returns:
            TokenBalance: User's token balance information.
        """
        async with self._get_connection() as db:
            cursor = await db.execute(
                "SELECT * FROM users WHERE api_key = ?",
                (api_key,)
            )
            row = await cursor.fetchone()
            
            if row is None:
                # Create new user
                now = datetime.now(timezone.utc).isoformat()
                await db.execute(
                    """
                    INSERT INTO users (api_key, balance, total_used, created_at)
                    VALUES (?, ?, 0, ?)
                    """,
                    (api_key, self.default_balance, now)
                )
                await db.commit()
                
                logger.info(
                    "New user created",
                    api_key=self._mask_key(api_key),
                    initial_balance=self.default_balance
                )
                
                return TokenBalance(
                    api_key=self._mask_key(api_key),
                    balance=self.default_balance,
                    total_used=0,
                    created_at=now,
                    last_used=None
                )
            
            return TokenBalance(
                api_key=self._mask_key(api_key),
                balance=row["balance"],
                total_used=row["total_used"],
                created_at=row["created_at"],
                last_used=row["last_used"]
            )
    
    async def get_balance(self, api_key: str) -> int:
        """
        Get current token balance for a user.
        
        Args:
            api_key: The user's API key.
            
        Returns:
            int: Current token balance.
        """
        user = await self.get_or_create_user(api_key)
        return user.balance
    
    async def has_sufficient_balance(
        self,
        api_key: str,
        required_tokens: int = 1
    ) -> bool:
        """
        Check if user has sufficient token balance.
        
        Args:
            api_key: The user's API key.
            required_tokens: Minimum required tokens.
            
        Returns:
            bool: True if balance is sufficient.
        """
        balance = await self.get_balance(api_key)
        return balance >= required_tokens
    
    async def deduct_tokens(
        self,
        api_key: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        request_id: str
    ) -> TokenDeductionResult:
        """
        Deduct tokens from user balance and log usage.
        
        Args:
            api_key: The user's API key.
            prompt_tokens: Number of prompt tokens used.
            completion_tokens: Number of completion tokens used.
            model: Model identifier.
            request_id: Unique request identifier.
            
        Returns:
            TokenDeductionResult: Result of the deduction operation.
        """
        total_tokens = prompt_tokens + completion_tokens
        now = datetime.now(timezone.utc).isoformat()
        
        async with self._get_connection() as db:
            # Get current balance
            cursor = await db.execute(
                "SELECT balance, total_used FROM users WHERE api_key = ?",
                (api_key,)
            )
            row = await cursor.fetchone()
            
            if row is None:
                # User doesn't exist, create with deduction
                new_balance = self.default_balance - total_tokens
                await db.execute(
                    """
                    INSERT INTO users (api_key, balance, total_used, created_at, last_used)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (api_key, new_balance, total_tokens, now, now)
                )
                previous_balance = self.default_balance
            else:
                previous_balance = row["balance"]
                new_balance = previous_balance - total_tokens
                
                # Update balance
                await db.execute(
                    """
                    UPDATE users 
                    SET balance = ?, total_used = total_used + ?, last_used = ?
                    WHERE api_key = ?
                    """,
                    (new_balance, total_tokens, now, api_key)
                )
            
            # Log usage
            await db.execute(
                """
                INSERT INTO usage_logs 
                (api_key, request_id, model, prompt_tokens, completion_tokens, total_tokens, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (api_key, request_id, model, prompt_tokens, completion_tokens, total_tokens, now)
            )
            
            await db.commit()
        
        logger.info(
            "Tokens deducted",
            api_key=self._mask_key(api_key),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            previous_balance=previous_balance,
            new_balance=new_balance,
            request_id=request_id
        )
        
        return TokenDeductionResult(
            success=True,
            previous_balance=previous_balance,
            current_balance=new_balance,
            tokens_deducted=total_tokens,
            message=f"Deducted {total_tokens} tokens"
        )
    
    async def add_tokens(self, api_key: str, amount: int) -> TokenBalance:
        """
        Add tokens to user balance.
        
        Args:
            api_key: The user's API key.
            amount: Number of tokens to add.
            
        Returns:
            TokenBalance: Updated balance information.
        """
        async with self._get_connection() as db:
            # Ensure user exists
            await self.get_or_create_user(api_key)
            
            # Add tokens
            await db.execute(
                "UPDATE users SET balance = balance + ? WHERE api_key = ?",
                (amount, api_key)
            )
            await db.commit()
        
        logger.info(
            "Tokens added",
            api_key=self._mask_key(api_key),
            amount=amount
        )
        
        return await self.get_or_create_user(api_key)
    
    async def get_usage_history(
        self,
        api_key: str,
        limit: int = 100
    ) -> list[TokenUsageLog]:
        """
        Get usage history for a user.
        
        Args:
            api_key: The user's API key.
            limit: Maximum number of records to return.
            
        Returns:
            list[TokenUsageLog]: List of usage log entries.
        """
        async with self._get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM usage_logs 
                WHERE api_key = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (api_key, limit)
            )
            rows = await cursor.fetchall()
        
        return [
            TokenUsageLog(
                api_key=self._mask_key(row["api_key"]),
                request_id=row["request_id"],
                model=row["model"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
                timestamp=row["timestamp"]
            )
            for row in rows
        ]
    
    @staticmethod
    def _mask_key(api_key: str) -> str:
        """Mask API key for logging."""
        if len(api_key) <= 8:
            return "*" * len(api_key)
        return f"{api_key[:4]}...{api_key[-4:]}"


# Global token manager instance
_token_manager: Optional[TokenManager] = None


async def get_token_manager() -> TokenManager:
    """Get the global token manager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
        await _token_manager.initialize()
    return _token_manager
