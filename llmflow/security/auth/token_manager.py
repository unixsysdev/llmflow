"""
Token Management Module

This module provides token management functionality for LLMFlow.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..interfaces.security import ISecurityProvider, Token, SecurityContext
from ..interfaces.security import AuthenticationError, SecurityError

logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    """Extended token information."""
    token: Token
    created_at: datetime
    last_used: datetime
    usage_count: int
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None


class TokenManager:
    """Manages token lifecycle and operations."""
    
    def __init__(self, security_provider: ISecurityProvider):
        self.security_provider = security_provider
        self.active_tokens: Dict[str, TokenInfo] = {}
        self.blacklisted_tokens: Set[str] = set()
        self.cleanup_interval = 300  # 5 minutes
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start token management services."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Token manager started")
    
    async def stop(self) -> None:
        """Stop token management services."""
        if not self._running:
            return
        
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Token manager stopped")
    
    async def create_token(self, credentials: Dict[str, Any], client_info: Dict[str, Any] = None) -> Token:
        """Create a new token."""
        try:
            # Generate token using security provider
            token = await self.security_provider.authenticate(credentials)
            
            if not token:
                raise AuthenticationError("Failed to create token")
            
            # Store token info
            token_info = TokenInfo(
                token=token,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                usage_count=0,
                client_ip=client_info.get("client_ip") if client_info else None,
                user_agent=client_info.get("user_agent") if client_info else None
            )
            
            self.active_tokens[token.token] = token_info
            
            logger.info(f"Created token for user {token.user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise
    
    async def validate_token(self, token_string: str) -> Optional[Token]:
        """Validate a token."""
        try:
            # Check if token is blacklisted
            if token_string in self.blacklisted_tokens:
                logger.warning(f"Attempted use of blacklisted token")
                return None
            
            # Check if token exists in our registry
            token_info = self.active_tokens.get(token_string)
            if not token_info:
                logger.warning(f"Unknown token attempted")
                return None
            
            # Check if token is expired
            if self._is_token_expired(token_info.token):
                await self._remove_token(token_string)
                logger.debug(f"Expired token removed")
                return None
            
            # Update usage info
            token_info.last_used = datetime.utcnow()
            token_info.usage_count += 1
            
            return token_info.token
            
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None
    
    async def refresh_token(self, refresh_token: str, client_info: Dict[str, Any] = None) -> Optional[Token]:
        """Refresh a token."""
        try:
            credentials = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }
            
            # Create new token
            new_token = await self.create_token(credentials, client_info)
            
            # Invalidate old token if it exists
            for token_string, token_info in list(self.active_tokens.items()):
                if token_info.token.refresh_token == refresh_token:
                    await self._remove_token(token_string)
                    break
            
            logger.info("Token refreshed successfully")
            return new_token
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None
    
    async def revoke_token(self, token_string: str) -> bool:
        """Revoke a specific token."""
        try:
            if token_string in self.active_tokens:
                token_info = self.active_tokens[token_string]
                await self._remove_token(token_string)
                logger.info(f"Revoked token for user {token_info.token.user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False
    
    async def revoke_user_tokens(self, user_id: str) -> int:
        """Revoke all tokens for a specific user."""
        try:
            revoked_count = 0
            tokens_to_revoke = []
            
            for token_string, token_info in self.active_tokens.items():
                if token_info.token.user_id == user_id:
                    tokens_to_revoke.append(token_string)
            
            for token_string in tokens_to_revoke:
                await self._remove_token(token_string)
                revoked_count += 1
            
            logger.info(f"Revoked {revoked_count} tokens for user {user_id}")
            return revoked_count
            
        except Exception as e:
            logger.error(f"User token revocation failed: {e}")
            return 0
    
    async def blacklist_token(self, token_string: str) -> bool:
        """Add token to blacklist."""
        try:
            self.blacklisted_tokens.add(token_string)
            await self._remove_token(token_string)
            logger.info("Token blacklisted")
            return True
            
        except Exception as e:
            logger.error(f"Token blacklisting failed: {e}")
            return False
    
    def get_token_info(self, token_string: str) -> Optional[TokenInfo]:
        """Get token information."""
        return self.active_tokens.get(token_string)
    
    def get_user_tokens(self, user_id: str) -> List[TokenInfo]:
        """Get all active tokens for a user."""
        return [
            token_info for token_info in self.active_tokens.values()
            if token_info.token.user_id == user_id
        ]
    
    def get_active_tokens_count(self) -> int:
        """Get count of active tokens."""
        return len(self.active_tokens)
    
    def get_blacklisted_tokens_count(self) -> int:
        """Get count of blacklisted tokens."""
        return len(self.blacklisted_tokens)
    
    async def get_token_stats(self) -> Dict[str, Any]:
        """Get token statistics."""
        now = datetime.utcnow()
        
        # Count tokens by expiration time
        expiring_soon = 0  # Within 1 hour
        expired = 0
        
        for token_info in self.active_tokens.values():
            if self._is_token_expired(token_info.token):
                expired += 1
            elif token_info.token.expires_at and (token_info.token.expires_at - now) < timedelta(hours=1):
                expiring_soon += 1
        
        # Count tokens by type
        token_types = {}
        for token_info in self.active_tokens.values():
            token_type = token_info.token.token_type
            token_types[token_type] = token_types.get(token_type, 0) + 1
        
        return {
            "active_tokens": len(self.active_tokens),
            "blacklisted_tokens": len(self.blacklisted_tokens),
            "expiring_soon": expiring_soon,
            "expired": expired,
            "token_types": token_types
        }
    
    # Internal helper methods
    def _is_token_expired(self, token: Token) -> bool:
        """Check if token is expired."""
        if not token.expires_at:
            return False
        return datetime.utcnow() > token.expires_at
    
    async def _remove_token(self, token_string: str) -> None:
        """Remove token from active tokens."""
        if token_string in self.active_tokens:
            del self.active_tokens[token_string]
    
    async def _cleanup_expired_tokens(self) -> int:
        """Remove expired tokens."""
        expired_tokens = []
        
        for token_string, token_info in self.active_tokens.items():
            if self._is_token_expired(token_info.token):
                expired_tokens.append(token_string)
        
        for token_string in expired_tokens:
            await self._remove_token(token_string)
        
        if expired_tokens:
            logger.debug(f"Cleaned up {len(expired_tokens)} expired tokens")
        
        return len(expired_tokens)
    
    async def _cleanup_old_blacklisted_tokens(self) -> int:
        """Clean up old blacklisted tokens."""
        # Keep blacklisted tokens for 7 days
        cutoff = datetime.utcnow() - timedelta(days=7)
        
        # For simplicity, we'll just limit the size of blacklist
        if len(self.blacklisted_tokens) > 10000:
            # Remove half of the tokens (oldest ones would be better, but this is simpler)
            tokens_to_remove = list(self.blacklisted_tokens)[:5000]
            for token in tokens_to_remove:
                self.blacklisted_tokens.remove(token)
            return 5000
        
        return 0
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup task."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Clean up expired tokens
                expired_count = await self._cleanup_expired_tokens()
                
                # Clean up old blacklisted tokens
                blacklist_cleaned = await self._cleanup_old_blacklisted_tokens()
                
                if expired_count > 0 or blacklist_cleaned > 0:
                    logger.info(f"Cleanup: {expired_count} expired tokens, {blacklist_cleaned} old blacklisted tokens")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Token cleanup error: {e}")


class TokenManagerRegistry:
    """Registry for multiple token managers."""
    
    def __init__(self):
        self.managers: Dict[str, TokenManager] = {}
        self.default_provider: Optional[str] = None
    
    def register_provider(self, name: str, security_provider: ISecurityProvider, is_default: bool = False) -> None:
        """Register a token manager for a security provider."""
        self.managers[name] = TokenManager(security_provider)
        
        if is_default or self.default_provider is None:
            self.default_provider = name
        
        logger.info(f"Registered token manager: {name}")
    
    def get_manager(self, provider_name: str = None) -> TokenManager:
        """Get token manager for a provider."""
        name = provider_name or self.default_provider
        
        if name not in self.managers:
            raise SecurityError(f"Token manager '{name}' not found")
        
        return self.managers[name]
    
    async def start_all(self) -> None:
        """Start all token managers."""
        for manager in self.managers.values():
            await manager.start()
        logger.info("All token managers started")
    
    async def stop_all(self) -> None:
        """Stop all token managers."""
        for manager in self.managers.values():
            await manager.stop()
        logger.info("All token managers stopped")
    
    async def create_token(self, credentials: Dict[str, Any], client_info: Dict[str, Any] = None, 
                          provider_name: str = None) -> Token:
        """Create token using specified or default provider."""
        manager = self.get_manager(provider_name)
        return await manager.create_token(credentials, client_info)
    
    async def validate_token(self, token_string: str, provider_name: str = None) -> Optional[Token]:
        """Validate token using specified or default provider."""
        manager = self.get_manager(provider_name)
        return await manager.validate_token(token_string)
    
    async def revoke_token(self, token_string: str, provider_name: str = None) -> bool:
        """Revoke token using specified or default provider."""
        manager = self.get_manager(provider_name)
        return await manager.revoke_token(token_string)


# Global token manager registry
_token_registry = TokenManagerRegistry()

def get_token_registry() -> TokenManagerRegistry:
    """Get the global token manager registry."""
    return _token_registry
