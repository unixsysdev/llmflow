"""
No Security Provider

This module provides a development-only security provider that bypasses authentication.
WARNING: This should only be used in development environments!
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..interfaces.security import ISecurityProvider, SecurityLevel, Token, SecurityContext
from ..interfaces.security import AuthenticationError, AuthorizationError, SecurityError
from ..interfaces.base import Plugin

logger = logging.getLogger(__name__)


class NoSecurityProvider(Plugin, ISecurityProvider):
    """Development-only security provider that bypasses authentication."""
    
    def __init__(self, config: Dict[str, Any] = None):
        Plugin.__init__(self, config)
        self.development_mode = self.config.get("development_mode", True)
        
        if not self.development_mode:
            raise SecurityError("NoSecurityProvider can only be used in development mode")
        
        logger.warning("⚠️  NoSecurityProvider is active - ALL AUTHENTICATION IS BYPASSED!")
    
    # Plugin interface methods
    def get_name(self) -> str:
        return "no_security_provider"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "Development-only provider that bypasses all security"
    
    def get_dependencies(self) -> List[str]:
        return []
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the no-security provider."""
        self.config.update(config)
        logger.warning("⚠️  No-security provider initialized - DEVELOPMENT MODE ONLY!")
    
    async def shutdown(self) -> None:
        """Shutdown the no-security provider."""
        logger.info("No-security provider shutdown")
    
    # Security provider interface methods
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Token]:
        """Always authenticate successfully (development only)."""
        logger.warning("⚠️  Authentication bypassed - returning dummy token")
        
        username = credentials.get("username", "dev_user")
        
        return Token(
            token="dev_token_12345",
            token_type="development",
            expires_at=datetime.utcnow() + timedelta(days=1),
            user_id=username
        )
    
    async def authorize(self, token: Token, resource: str, action: str) -> bool:
        """Always authorize successfully (development only)."""
        logger.warning(f"⚠️  Authorization bypassed for {resource}:{action}")
        return True
    
    async def encrypt(self, data: bytes, key: str) -> bytes:
        """Return data unchanged (no encryption in development)."""
        logger.warning("⚠️  Encryption bypassed - data returned unchanged")
        return data
    
    async def decrypt(self, data: bytes, key: str) -> bytes:
        """Return data unchanged (no decryption in development)."""
        logger.warning("⚠️  Decryption bypassed - data returned unchanged")
        return data
    
    async def sign(self, data: bytes, key: str) -> bytes:
        """Return dummy signature."""
        logger.warning("⚠️  Signing bypassed - returning dummy signature")
        return b"dev_signature"
    
    async def verify(self, data: bytes, signature: bytes, key: str) -> bool:
        """Always verify successfully."""
        logger.warning("⚠️  Signature verification bypassed - always returns True")
        return True
    
    async def generate_key(self) -> str:
        """Generate a development key."""
        return "dev_key_12345"
    
    async def create_security_context(self, token: Token) -> SecurityContext:
        """Create a development security context."""
        return SecurityContext(
            user_id=token.user_id or "dev_user",
            permissions=["admin:*"],  # Grant all permissions in development
            security_level=SecurityLevel.AUTHENTICATED,
            domain="development",
            tenant_id="dev_tenant",
            expires_at=token.expires_at
        )
