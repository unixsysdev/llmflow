"""
Example Security Plugin - Dummy Security Provider

This is a simple example plugin that demonstrates how to implement
the ISecurityProvider interface for LLMFlow.
"""

import logging
import hashlib
import hmac
import base64
import json
from typing import Any, Dict, List, Optional, Type
from datetime import datetime, timedelta

from ..interfaces.base import Plugin, PluginStatus
from ..interfaces.security import (
    ISecurityProvider, AuthenticationMethod, SecurityError, 
    AuthenticationError, AuthorizationError, CryptographyError,
    Token, SecurityContext, SecurityLevel
)

logger = logging.getLogger(__name__)


class DummySecurityProvider(Plugin, ISecurityProvider):
    """
    Dummy security provider plugin for testing and demonstration.
    
    This plugin implements a basic security provider that can be used
    for testing the plugin system without requiring actual cryptographic operations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        Plugin.__init__(self, config)
        self.users: Dict[str, Dict[str, Any]] = {}
        self.tokens: Dict[str, Token] = {}
        self.keys: Dict[str, str] = {}
        self.secret_key = "dummy-secret-key"
        self.audit_logs: List[Dict[str, Any]] = []
    
    # Plugin interface methods
    def get_name(self) -> str:
        return "dummy_security_provider"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "Dummy security provider plugin for testing and demonstration"
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def get_interfaces(self) -> List[Type]:
        return [ISecurityProvider]
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the security plugin."""
        self.status = PluginStatus.INITIALIZING
        try:
            self.config.update(config)
            self.secret_key = self.config.get('secret_key', 'dummy-secret-key')
            
            # Create some dummy users
            self.users = {
                'admin': {
                    'password': self._hash_password('admin123'),
                    'permissions': ['read', 'write', 'admin'],
                    'security_level': SecurityLevel.TOP_SECRET
                },
                'user': {
                    'password': self._hash_password('user123'),
                    'permissions': ['read'],
                    'security_level': SecurityLevel.RESTRICTED
                },
                'guest': {
                    'password': self._hash_password('guest123'),
                    'permissions': [],
                    'security_level': SecurityLevel.PUBLIC
                }
            }
            
            self.status = PluginStatus.INITIALIZED
            logger.info("Dummy security provider initialized")
        except Exception as e:
            self.status = PluginStatus.ERROR
            raise e
    
    async def start(self) -> None:
        """Start the security plugin."""
        self.status = PluginStatus.STARTING
        try:
            self.status = PluginStatus.RUNNING
            logger.info("Dummy security provider started")
        except Exception as e:
            self.status = PluginStatus.ERROR
            raise e
    
    async def stop(self) -> None:
        """Stop the security plugin."""
        self.status = PluginStatus.STOPPING
        try:
            self.status = PluginStatus.STOPPED
            logger.info("Dummy security provider stopped")
        except Exception as e:
            self.status = PluginStatus.ERROR
            raise e
    
    async def shutdown(self) -> None:
        """Shutdown the security plugin."""
        await self.stop()
        logger.info("Dummy security provider shutdown")
    
    async def health_check(self) -> bool:
        """Check if the security provider is healthy."""
        return self.status == PluginStatus.RUNNING
    
    # Security interface methods
    def get_authentication_method(self) -> AuthenticationMethod:
        return AuthenticationMethod.PASSWORD
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Token]:
        """Authenticate using the provided credentials."""
        try:
            username = credentials.get('username')
            password = credentials.get('password')
            
            if not username or not password:
                raise AuthenticationError("Username and password required")
            
            # Check if user exists
            if username not in self.users:
                raise AuthenticationError("Invalid username or password")
            
            # Verify password
            user = self.users[username]
            if not self._verify_password(password, user['password']):
                raise AuthenticationError("Invalid username or password")
            
            # Create token
            token_data = {
                'username': username,
                'permissions': user['permissions'],
                'security_level': user['security_level'].value,
                'issued_at': datetime.utcnow().isoformat()
            }
            
            token_string = self._create_token(token_data)
            expires_at = datetime.utcnow() + timedelta(hours=1)
            
            token = Token(
                token=token_string,
                expires_at=expires_at,
                scope=user['permissions'],
                metadata={'username': username}
            )
            
            self.tokens[token_string] = token
            
            await self.audit_log("authenticate", None, username, True)
            logger.info(f"User {username} authenticated successfully")
            return token
            
        except Exception as e:
            await self.audit_log("authenticate", None, username, False)
            logger.error(f"Authentication failed: {e}")
            raise AuthenticationError(str(e))
    
    async def validate_token(self, token: str) -> Optional[Token]:
        """Validate an authentication token."""
        try:
            if token not in self.tokens:
                return None
            
            token_obj = self.tokens[token]
            if token_obj.is_expired():
                del self.tokens[token]
                return None
            
            return token_obj
            
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None
    
    async def refresh_token(self, token: Token) -> Optional[Token]:
        """Refresh an authentication token."""
        try:
            if token.token not in self.tokens:
                return None
            
            # Create new token with same permissions
            new_token_data = {
                'username': token.metadata.get('username'),
                'permissions': token.scope,
                'security_level': token.metadata.get('security_level', 'public'),
                'issued_at': datetime.utcnow().isoformat()
            }
            
            new_token_string = self._create_token(new_token_data)
            expires_at = datetime.utcnow() + timedelta(hours=1)
            
            new_token = Token(
                token=new_token_string,
                expires_at=expires_at,
                scope=token.scope,
                metadata=token.metadata
            )
            
            # Remove old token and add new one
            del self.tokens[token.token]
            self.tokens[new_token_string] = new_token
            
            logger.info(f"Token refreshed for user {token.metadata.get('username')}")
            return new_token
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None
    
    async def revoke_token(self, token: Token) -> bool:
        """Revoke an authentication token."""
        try:
            if token.token in self.tokens:
                del self.tokens[token.token]
                logger.info(f"Token revoked for user {token.metadata.get('username')}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False
    
    async def authorize(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Authorize an action on a resource."""
        try:
            if not context.is_authenticated():
                return False
            
            # Check if user has required permission
            required_permission = f"{action}:{resource}"
            if required_permission in context.permissions:
                await self.audit_log("authorize", context, resource, True)
                return True
            
            # Check for wildcard permissions
            if f"{action}:*" in context.permissions or f"*:{resource}" in context.permissions:
                await self.audit_log("authorize", context, resource, True)
                return True
            
            # Check for admin permission
            if "admin" in context.permissions:
                await self.audit_log("authorize", context, resource, True)
                return True
            
            await self.audit_log("authorize", context, resource, False)
            return False
            
        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            await self.audit_log("authorize", context, resource, False)
            return False
    
    async def encrypt(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        """Encrypt data using the specified key."""
        try:
            # Simple XOR encryption for demonstration
            key = self.secret_key.encode('utf-8')
            encrypted = bytearray()
            
            for i, byte in enumerate(data):
                encrypted.append(byte ^ key[i % len(key)])
            
            return bytes(encrypted)
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise CryptographyError(str(e))
    
    async def decrypt(self, encrypted_data: bytes, key_id: Optional[str] = None) -> bytes:
        """Decrypt data using the specified key."""
        try:
            # Simple XOR decryption for demonstration
            key = self.secret_key.encode('utf-8')
            decrypted = bytearray()
            
            for i, byte in enumerate(encrypted_data):
                decrypted.append(byte ^ key[i % len(key)])
            
            return bytes(decrypted)
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise CryptographyError(str(e))
    
    async def sign(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        """Sign data using the specified key."""
        try:
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                data,
                hashlib.sha256
            ).digest()
            
            return signature
            
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            raise CryptographyError(str(e))
    
    async def verify(self, data: bytes, signature: bytes, key_id: Optional[str] = None) -> bool:
        """Verify a signature for the given data."""
        try:
            expected_signature = hmac.new(
                self.secret_key.encode('utf-8'),
                data,
                hashlib.sha256
            ).digest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    async def generate_key(self, key_type: str = "rsa", key_size: int = 2048) -> str:
        """Generate a new cryptographic key."""
        try:
            # Generate a simple key for demonstration
            import uuid
            key_id = str(uuid.uuid4())
            self.keys[key_id] = f"dummy-key-{key_id}"
            
            logger.info(f"Generated new key: {key_id}")
            return key_id
            
        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            raise CryptographyError(str(e))
    
    async def rotate_key(self, key_id: str) -> str:
        """Rotate a cryptographic key."""
        try:
            if key_id not in self.keys:
                raise CryptographyError("Key not found")
            
            # Generate new key
            new_key_id = await self.generate_key()
            
            # Remove old key
            del self.keys[key_id]
            
            logger.info(f"Key rotated: {key_id} -> {new_key_id}")
            return new_key_id
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise CryptographyError(str(e))
    
    async def create_security_context(self, user_id: str, token: Optional[Token] = None) -> SecurityContext:
        """Create a security context for a user."""
        try:
            if user_id not in self.users:
                raise SecurityError("User not found")
            
            user = self.users[user_id]
            
            context = SecurityContext(
                user_id=user_id,
                token=token,
                security_level=user['security_level'],
                permissions=user['permissions']
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Security context creation failed: {e}")
            raise SecurityError(str(e))
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get permissions for a user."""
        try:
            if user_id not in self.users:
                return []
            
            return self.users[user_id]['permissions']
            
        except Exception as e:
            logger.error(f"Getting user permissions failed: {e}")
            return []
    
    async def audit_log(self, action: str, context: Optional[SecurityContext], resource: str, success: bool) -> None:
        """Log a security event for auditing."""
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': action,
                'user_id': context.user_id if context else None,
                'resource': resource,
                'success': success
            }
            
            self.audit_logs.append(log_entry)
            logger.info(f"Audit log: {log_entry}")
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
    
    # Helper methods
    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return self._hash_password(password) == hashed
    
    def _create_token(self, data: Dict[str, Any]) -> str:
        """Create a simple token."""
        payload = base64.b64encode(json.dumps(data).encode('utf-8')).decode('utf-8')
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"{payload}.{signature}"
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log."""
        return self.audit_logs.copy()
    
    def get_users(self) -> List[str]:
        """Get list of users."""
        return list(self.users.keys())
    
    def get_active_tokens(self) -> int:
        """Get number of active tokens."""
        return len(self.tokens)
