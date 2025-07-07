"""
Security Provider Interface

This module defines the interface for security providers in the LLMFlow framework.
Security providers handle authentication, authorization, encryption, and signing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for contexts and operations."""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class AuthenticationMethod(Enum):
    """Authentication methods."""
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    OAUTH2 = "oauth2"
    LDAP = "ldap"
    NONE = "none"


class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    pass


class CryptographyError(SecurityError):
    """Raised when cryptographic operations fail."""
    pass


class Token:
    """
    Represents an authentication token.
    """
    
    def __init__(self, 
                 token: str, 
                 token_type: str = "Bearer",
                 expires_at: Optional[datetime] = None,
                 scope: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.token = token
        self.token_type = token_type
        self.expires_at = expires_at
        self.scope = scope or []
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if the token is valid (not expired)."""
        return not self.is_expired()
    
    def has_scope(self, scope: str) -> bool:
        """Check if the token has the required scope."""
        return scope in self.scope
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert token to dictionary."""
        return {
            'token': self.token,
            'token_type': self.token_type,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'scope': self.scope,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class SecurityContext:
    """
    Security context for operations.
    """
    
    def __init__(self, 
                 user_id: Optional[str] = None,
                 token: Optional[Token] = None,
                 security_level: SecurityLevel = SecurityLevel.PUBLIC,
                 permissions: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.user_id = user_id
        self.token = token
        self.security_level = security_level
        self.permissions = permissions or []
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def has_permission(self, permission: str) -> bool:
        """Check if the context has the required permission."""
        return permission in self.permissions
    
    def is_authenticated(self) -> bool:
        """Check if the context is authenticated."""
        return self.user_id is not None and (self.token is None or self.token.is_valid())
    
    def can_access_level(self, required_level: SecurityLevel) -> bool:
        """Check if the context can access the required security level."""
        levels = [SecurityLevel.PUBLIC, SecurityLevel.RESTRICTED, SecurityLevel.CONFIDENTIAL, 
                 SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]
        return levels.index(self.security_level) >= levels.index(required_level)


class ISecurityProvider(ABC):
    """
    Interface for security providers in LLMFlow.
    
    This interface defines the contract for all security implementations,
    including authentication, authorization, encryption, and signing.
    """
    
    @abstractmethod
    def get_authentication_method(self) -> AuthenticationMethod:
        """
        Get the authentication method this provider supports.
        
        Returns:
            The authentication method
        """
        pass
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Token]:
        """
        Authenticate using the provided credentials.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            Token if authentication successful, None otherwise
            
        Raises:
            AuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[Token]:
        """
        Validate an authentication token.
        
        Args:
            token: Token to validate
            
        Returns:
            Token object if valid, None otherwise
            
        Raises:
            AuthenticationError: If token validation fails
        """
        pass
    
    @abstractmethod
    async def refresh_token(self, token: Token) -> Optional[Token]:
        """
        Refresh an authentication token.
        
        Args:
            token: Token to refresh
            
        Returns:
            New token if refresh successful, None otherwise
            
        Raises:
            AuthenticationError: If token refresh fails
        """
        pass
    
    @abstractmethod
    async def revoke_token(self, token: Token) -> bool:
        """
        Revoke an authentication token.
        
        Args:
            token: Token to revoke
            
        Returns:
            True if revocation successful, False otherwise
            
        Raises:
            AuthenticationError: If token revocation fails
        """
        pass
    
    @abstractmethod
    async def authorize(self, context: SecurityContext, resource: str, action: str) -> bool:
        """
        Authorize an action on a resource.
        
        Args:
            context: Security context
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            True if authorized, False otherwise
            
        Raises:
            AuthorizationError: If authorization check fails
        """
        pass
    
    @abstractmethod
    async def encrypt(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        """
        Encrypt data using the specified key.
        
        Args:
            data: Data to encrypt
            key_id: Optional key identifier
            
        Returns:
            Encrypted data
            
        Raises:
            CryptographyError: If encryption fails
        """
        pass
    
    @abstractmethod
    async def decrypt(self, encrypted_data: bytes, key_id: Optional[str] = None) -> bytes:
        """
        Decrypt data using the specified key.
        
        Args:
            encrypted_data: Data to decrypt
            key_id: Optional key identifier
            
        Returns:
            Decrypted data
            
        Raises:
            CryptographyError: If decryption fails
        """
        pass
    
    @abstractmethod
    async def sign(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        """
        Sign data using the specified key.
        
        Args:
            data: Data to sign
            key_id: Optional key identifier
            
        Returns:
            Signature
            
        Raises:
            CryptographyError: If signing fails
        """
        pass
    
    @abstractmethod
    async def verify(self, data: bytes, signature: bytes, key_id: Optional[str] = None) -> bool:
        """
        Verify a signature for the given data.
        
        Args:
            data: Original data
            signature: Signature to verify
            key_id: Optional key identifier
            
        Returns:
            True if signature is valid, False otherwise
            
        Raises:
            CryptographyError: If signature verification fails
        """
        pass
    
    @abstractmethod
    async def generate_key(self, key_type: str = "rsa", key_size: int = 2048) -> str:
        """
        Generate a new cryptographic key.
        
        Args:
            key_type: Type of key to generate
            key_size: Size of key in bits
            
        Returns:
            Key identifier
            
        Raises:
            CryptographyError: If key generation fails
        """
        pass
    
    @abstractmethod
    async def rotate_key(self, key_id: str) -> str:
        """
        Rotate a cryptographic key.
        
        Args:
            key_id: Key to rotate
            
        Returns:
            New key identifier
            
        Raises:
            CryptographyError: If key rotation fails
        """
        pass
    
    @abstractmethod
    async def create_security_context(self, user_id: str, token: Optional[Token] = None) -> SecurityContext:
        """
        Create a security context for a user.
        
        Args:
            user_id: User identifier
            token: Optional authentication token
            
        Returns:
            Security context
            
        Raises:
            SecurityError: If context creation fails
        """
        pass
    
    @abstractmethod
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """
        Get permissions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of permissions
            
        Raises:
            SecurityError: If permission retrieval fails
        """
        pass
    
    @abstractmethod
    async def audit_log(self, action: str, context: SecurityContext, resource: str, success: bool) -> None:
        """
        Log a security event for auditing.
        
        Args:
            action: Action performed
            context: Security context
            resource: Resource accessed
            success: Whether the action was successful
            
        Raises:
            SecurityError: If audit logging fails
        """
        pass
