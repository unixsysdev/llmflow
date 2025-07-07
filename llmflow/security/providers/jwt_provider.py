"""
JWT Security Provider

This module provides JWT-based authentication and authorization for LLMFlow.
"""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..interfaces.security import ISecurityProvider, SecurityLevel, Token, SecurityContext
from ..interfaces.security import AuthenticationError, AuthorizationError, SecurityError
from ..interfaces.base import Plugin

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    API = "api"


@dataclass
class JWTConfig:
    """Configuration for JWT provider."""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    api_token_expire_hours: int = 24
    issuer: str = "llmflow"
    audience: str = "llmflow-api"


class JWTSecurityProvider(Plugin, ISecurityProvider):
    """JWT-based security provider."""
    
    def __init__(self, config: Dict[str, Any] = None):
        Plugin.__init__(self, config)
        self.jwt_config = JWTConfig(**self.config)
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate JWT configuration."""
        if not self.jwt_config.secret_key:
            raise SecurityError("JWT secret key is required")
        
        if len(self.jwt_config.secret_key) < 32:
            raise SecurityError("JWT secret key must be at least 32 characters")
    
    # Plugin interface methods
    def get_name(self) -> str:
        return "jwt_security_provider"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "JWT-based authentication and authorization provider"
    
    def get_dependencies(self) -> List[str]:
        return ["PyJWT"]
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the JWT provider."""
        self.config.update(config)
        self.jwt_config = JWTConfig(**self.config)
        self._validate_config()
        logger.info("JWT security provider initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the JWT provider."""
        logger.info("JWT security provider shutdown")
    
    # Security provider interface methods
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Token]:
        """Authenticate user credentials and return a token."""
        try:
            # Extract credentials
            username = credentials.get("username")
            password = credentials.get("password")
            token_type = credentials.get("token_type", TokenType.ACCESS.value)
            
            if not username or not password:
                raise AuthenticationError("Username and password are required")
            
            # Validate credentials (simplified - in production would check against database)
            if not await self._validate_credentials(username, password):
                raise AuthenticationError("Invalid credentials")
            
            # Generate token
            token = await self._generate_token(username, TokenType(token_type))
            logger.info(f"User {username} authenticated successfully")
            return token
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            if isinstance(e, AuthenticationError):
                raise
            raise AuthenticationError(f"Authentication failed: {e}")
    
    async def authorize(self, token: Token, resource: str, action: str) -> bool:
        """Authorize access to a resource."""
        try:
            # Verify token
            payload = await self._verify_token(token)
            username = payload.get("sub")
            
            # Check permissions (simplified - in production would use RBAC)
            permissions = await self._get_user_permissions(username)
            required_permission = f"{resource}:{action}"
            
            authorized = required_permission in permissions or "admin:*" in permissions
            
            if authorized:
                logger.debug(f"User {username} authorized for {required_permission}")
            else:
                logger.warning(f"User {username} denied access to {required_permission}")
            
            return authorized
            
        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            if isinstance(e, AuthorizationError):
                raise
            raise AuthorizationError(f"Authorization failed: {e}")
    
    async def encrypt(self, data: bytes, key: str) -> bytes:
        """Encrypt data (not implemented in JWT provider)."""
        raise NotImplementedError("Encryption not supported by JWT provider")
    
    async def decrypt(self, data: bytes, key: str) -> bytes:
        """Decrypt data (not implemented in JWT provider)."""
        raise NotImplementedError("Decryption not supported by JWT provider")
    
    async def sign(self, data: bytes, key: str) -> bytes:
        """Sign data using JWT."""
        try:
            # Create JWT with data as payload
            payload = {
                "data": data.hex(),
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(hours=1),
                "iss": self.jwt_config.issuer
            }
            
            token = jwt.encode(payload, key or self.jwt_config.secret_key, algorithm=self.jwt_config.algorithm)
            return token.encode()
            
        except Exception as e:
            raise SecurityError(f"Signing failed: {e}")
    
    async def verify(self, data: bytes, signature: bytes, key: str) -> bool:
        """Verify JWT signature."""
        try:
            token = signature.decode()
            payload = jwt.decode(
                token, 
                key or self.jwt_config.secret_key, 
                algorithms=[self.jwt_config.algorithm],
                issuer=self.jwt_config.issuer
            )
            
            # Verify data matches
            signed_data = bytes.fromhex(payload["data"])
            return signed_data == data
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    async def generate_key(self) -> str:
        """Generate a new JWT secret key."""
        import secrets
        return secrets.token_urlsafe(32)
    
    async def create_security_context(self, token: Token) -> SecurityContext:
        """Create security context from token."""
        try:
            payload = await self._verify_token(token)
            
            return SecurityContext(
                user_id=payload.get("sub"),
                permissions=await self._get_user_permissions(payload.get("sub")),
                security_level=SecurityLevel.AUTHENTICATED,
                domain=payload.get("domain", "default"),
                tenant_id=payload.get("tenant_id", "default"),
                expires_at=datetime.fromtimestamp(payload.get("exp"))
            )
            
        except Exception as e:
            raise SecurityError(f"Failed to create security context: {e}")
    
    # Internal helper methods
    async def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials."""
        # Simplified validation - in production would check against database/LDAP
        # For demo purposes, accept admin/admin and user/user
        valid_users = {
            "admin": "admin",
            "user": "user", 
            "test": "test"
        }
        
        return username in valid_users and valid_users[username] == password
    
    async def _generate_token(self, username: str, token_type: TokenType) -> Token:
        """Generate JWT token."""
        now = datetime.utcnow()
        
        # Set expiration based on token type
        if token_type == TokenType.ACCESS:
            expires_delta = timedelta(minutes=self.jwt_config.access_token_expire_minutes)
        elif token_type == TokenType.REFRESH:
            expires_delta = timedelta(days=self.jwt_config.refresh_token_expire_days)
        elif token_type == TokenType.API:
            expires_delta = timedelta(hours=self.jwt_config.api_token_expire_hours)
        else:
            expires_delta = timedelta(minutes=self.jwt_config.access_token_expire_minutes)
        
        payload = {
            "sub": username,
            "iat": now,
            "exp": now + expires_delta,
            "iss": self.jwt_config.issuer,
            "aud": self.jwt_config.audience,
            "token_type": token_type.value,
            "domain": "default",
            "tenant_id": "default"
        }
        
        token_string = jwt.encode(payload, self.jwt_config.secret_key, algorithm=self.jwt_config.algorithm)
        
        return Token(
            token=token_string,
            token_type=token_type.value,
            expires_at=now + expires_delta,
            user_id=username
        )
    
    async def _verify_token(self, token: Token) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token.token,
                self.jwt_config.secret_key,
                algorithms=[self.jwt_config.algorithm],
                issuer=self.jwt_config.issuer,
                audience=self.jwt_config.audience
            )
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    async def _get_user_permissions(self, username: str) -> List[str]:
        """Get user permissions."""
        # Simplified permissions - in production would come from database/RBAC
        permission_map = {
            "admin": ["admin:*", "queue:*", "transport:*", "security:*"],
            "user": ["queue:read", "queue:write", "transport:send", "transport:receive"],
            "test": ["queue:read", "transport:receive"]
        }
        
        return permission_map.get(username, [])
