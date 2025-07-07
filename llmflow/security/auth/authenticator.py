"""
Authentication Module

This module provides authentication logic for LLMFlow.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from ..interfaces.security import ISecurityProvider, Token, SecurityContext
from ..interfaces.security import AuthenticationError, SecurityError

logger = logging.getLogger(__name__)


class Authenticator:
    """Handles authentication operations."""
    
    def __init__(self, security_provider: ISecurityProvider):
        self.security_provider = security_provider
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Token:
        """Authenticate user with credentials."""
        username = credentials.get("username")
        
        try:
            # Check if user is locked out
            if await self._is_locked_out(username):
                raise AuthenticationError(f"Account {username} is temporarily locked due to too many failed attempts")
            
            # Attempt authentication
            token = await self.security_provider.authenticate(credentials)
            
            if token:
                # Clear failed attempts on successful authentication
                if username in self.failed_attempts:
                    del self.failed_attempts[username]
                
                # Create security context
                security_context = await self.security_provider.create_security_context(token)
                self.active_sessions[token.token] = security_context
                
                logger.info(f"User {username} authenticated successfully")
                return token
            else:
                await self._record_failed_attempt(username)
                raise AuthenticationError("Authentication failed")
                
        except AuthenticationError:
            await self._record_failed_attempt(username)
            raise
        except Exception as e:
            await self._record_failed_attempt(username)
            logger.error(f"Authentication error for {username}: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")
    
    async def get_security_context(self, token: str) -> Optional[SecurityContext]:
        """Get security context for a token."""
        return self.active_sessions.get(token)
    
    async def invalidate_token(self, token: str) -> bool:
        """Invalidate a token (logout)."""
        if token in self.active_sessions:
            context = self.active_sessions[token]
            del self.active_sessions[token]
            logger.info(f"Token invalidated for user {context.user_id}")
            return True
        return False
    
    async def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        invalidated = 0
        tokens_to_remove = []
        
        for token, context in self.active_sessions.items():
            if context.user_id == user_id:
                tokens_to_remove.append(token)
        
        for token in tokens_to_remove:
            del self.active_sessions[token]
            invalidated += 1
        
        logger.info(f"Invalidated {invalidated} sessions for user {user_id}")
        return invalidated
    
    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        now = datetime.utcnow()
        expired_tokens = []
        
        for token, context in self.active_sessions.items():
            if context.expires_at and context.expires_at < now:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            context = self.active_sessions[token]
            del self.active_sessions[token]
            logger.debug(f"Expired session removed for user {context.user_id}")
        
        return len(expired_tokens)
    
    async def refresh_token(self, refresh_token: str) -> Token:
        """Refresh an authentication token."""
        try:
            credentials = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }
            
            token = await self.security_provider.authenticate(credentials)
            
            if token:
                # Create new security context
                security_context = await self.security_provider.create_security_context(token)
                self.active_sessions[token.token] = security_context
                
                logger.info("Token refreshed successfully")
                return token
            else:
                raise AuthenticationError("Token refresh failed")
                
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            raise AuthenticationError(f"Token refresh failed: {e}")
    
    async def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        return len(self.active_sessions)
    
    async def get_user_sessions(self, user_id: str) -> List[SecurityContext]:
        """Get all active sessions for a user."""
        return [context for context in self.active_sessions.values() if context.user_id == user_id]
    
    # Internal helper methods
    async def _is_locked_out(self, username: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        if username not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[username]
        if len(attempts) < self.max_failed_attempts:
            return False
        
        # Check if lockout period has expired
        now = datetime.utcnow()
        latest_attempt = max(attempts)
        lockout_expires = latest_attempt + timedelta(minutes=self.lockout_duration_minutes)
        
        if now > lockout_expires:
            # Lockout expired, clear attempts
            del self.failed_attempts[username]
            return False
        
        return True
    
    async def _record_failed_attempt(self, username: str) -> None:
        """Record a failed authentication attempt."""
        if not username:
            return
        
        now = datetime.utcnow()
        
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(now)
        
        # Keep only recent attempts (within lockout window)
        cutoff = now - timedelta(minutes=self.lockout_duration_minutes)
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username] 
            if attempt > cutoff
        ]
        
        attempt_count = len(self.failed_attempts[username])
        logger.warning(f"Failed authentication attempt {attempt_count} for user {username}")


class AuthenticationManager:
    """High-level authentication manager."""
    
    def __init__(self):
        self.authenticators: Dict[str, Authenticator] = {}
        self.default_provider: Optional[str] = None
    
    def register_provider(self, name: str, security_provider: ISecurityProvider, is_default: bool = False) -> None:
        """Register a security provider."""
        self.authenticators[name] = Authenticator(security_provider)
        
        if is_default or self.default_provider is None:
            self.default_provider = name
        
        logger.info(f"Registered security provider: {name}")
    
    def get_authenticator(self, provider_name: str = None) -> Authenticator:
        """Get authenticator for a provider."""
        name = provider_name or self.default_provider
        
        if name not in self.authenticators:
            raise SecurityError(f"Security provider '{name}' not found")
        
        return self.authenticators[name]
    
    async def authenticate(self, credentials: Dict[str, Any], provider_name: str = None) -> Token:
        """Authenticate using specified or default provider."""
        authenticator = self.get_authenticator(provider_name)
        return await authenticator.authenticate(credentials)
    
    async def get_security_context(self, token: str, provider_name: str = None) -> Optional[SecurityContext]:
        """Get security context for a token."""
        authenticator = self.get_authenticator(provider_name)
        return await authenticator.get_security_context(token)
    
    async def cleanup_all_expired_sessions(self) -> int:
        """Cleanup expired sessions across all providers."""
        total_cleaned = 0
        for authenticator in self.authenticators.values():
            total_cleaned += await authenticator.cleanup_expired_sessions()
        return total_cleaned


# Global authentication manager instance
_auth_manager = AuthenticationManager()

def get_auth_manager() -> AuthenticationManager:
    """Get the global authentication manager."""
    return _auth_manager
