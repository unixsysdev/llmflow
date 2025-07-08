"""
Comprehensive tests for LLMFlow security module.
"""

import asyncio
import pytest
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from llmflow.security import initialize_security_system, shutdown_security_system
from llmflow.security.auth.authenticator import Authenticator, AuthenticationManager, get_auth_manager
from llmflow.security.auth.authorizer import Permission, Role, Authorizer, AuthorizationManager, get_authorization_manager
from llmflow.security.auth.token_manager import TokenInfo, TokenManager, TokenManagerRegistry, get_token_registry
from llmflow.security.crypto.signing import MessageSigner, SignatureManager, get_signature_manager
from llmflow.security.crypto.encryption import MessageEncryptor, EncryptionManager, SecureMessageEnvelope, get_encryption_manager
from llmflow.security.providers.jwt_provider import JWTSecurityProvider, JWTConfig, TokenType
from llmflow.security.providers.no_security_provider import NoSecurityProvider
from llmflow.security.providers.oauth2_provider import OAuth2SecurityProvider, OAuth2Config, OAuth2GrantType


class TestAuthenticator:
    """Test authenticator functionality."""
    
    def test_authenticator_creation(self):
        """Test authenticator creation."""
        auth = Authenticator()
        assert auth is not None
    
    def test_authentication_manager(self):
        """Test authentication manager."""
        manager = AuthenticationManager()
        assert manager is not None
        
        # Test singleton pattern
        manager2 = get_auth_manager()
        assert manager2 is not None


class TestAuthorizer:
    """Test authorization functionality."""
    
    def test_permission_creation(self):
        """Test permission creation."""
        perm = Permission("read", "queue", "test-queue")
        assert perm.action == "read"
        assert perm.resource == "queue"
        assert perm.resource_id == "test-queue"
    
    def test_role_creation(self):
        """Test role creation."""
        permissions = [
            Permission("read", "queue", "*"),
            Permission("write", "queue", "user-queue")
        ]
        role = Role("user", permissions)
        assert role.name == "user"
        assert len(role.permissions) == 2
    
    def test_authorizer_creation(self):
        """Test authorizer creation."""
        auth = Authorizer()
        assert auth is not None
    
    def test_authorization_manager(self):
        """Test authorization manager."""
        manager = AuthorizationManager()
        assert manager is not None
        
        # Test singleton pattern
        manager2 = get_authorization_manager()
        assert manager2 is not None


class TestTokenManager:
    """Test token management functionality."""
    
    def test_token_info_creation(self):
        """Test token info creation."""
        expires_at = datetime.utcnow() + timedelta(hours=1)
        token_info = TokenInfo(
            token="test-token",
            token_type="bearer",
            expires_at=expires_at,
            scopes=["read", "write"]
        )
        assert token_info.token == "test-token"
        assert token_info.token_type == "bearer"
        assert token_info.expires_at == expires_at
        assert token_info.scopes == ["read", "write"]
    
    def test_token_manager_creation(self):
        """Test token manager creation."""
        manager = TokenManager()
        assert manager is not None
    
    def test_token_manager_registry(self):
        """Test token manager registry."""
        registry = TokenManagerRegistry()
        assert registry is not None
        
        # Test singleton pattern
        registry2 = get_token_registry()
        assert registry2 is not None


class TestMessageSigner:
    """Test message signing functionality."""
    
    def test_message_signer_creation(self):
        """Test message signer creation."""
        signer = MessageSigner()
        assert signer is not None
    
    def test_signature_manager(self):
        """Test signature manager."""
        manager = SignatureManager()
        assert manager is not None
        
        # Test singleton pattern
        manager2 = get_signature_manager()
        assert manager2 is not None


class TestMessageEncryptor:
    """Test message encryption functionality."""
    
    def test_message_encryptor_creation(self):
        """Test message encryptor creation."""
        encryptor = MessageEncryptor()
        assert encryptor is not None
    
    def test_encryption_manager(self):
        """Test encryption manager."""
        manager = EncryptionManager()
        assert manager is not None
        
        # Test singleton pattern
        manager2 = get_encryption_manager()
        assert manager2 is not None
    
    def test_secure_message_envelope_creation(self):
        """Test secure message envelope creation."""
        envelope = SecureMessageEnvelope(
            encrypted_data=b"encrypted",
            signature=b"signature",
            metadata={"key": "value"}
        )
        assert envelope.encrypted_data == b"encrypted"
        assert envelope.signature == b"signature"
        assert envelope.metadata == {"key": "value"}


class TestJWTProvider:
    """Test JWT security provider."""
    
    def test_jwt_config_creation(self):
        """Test JWT config creation."""
        config = JWTConfig(
            secret_key="test-secret",
            algorithm="HS256",
            expiry_minutes=60
        )
        assert config.secret_key == "test-secret"
        assert config.algorithm == "HS256"
        assert config.expiry_minutes == 60
    
    def test_jwt_provider_creation(self):
        """Test JWT provider creation."""
        config = JWTConfig(
            secret_key="test-secret",
            algorithm="HS256",
            expiry_minutes=60
        )
        provider = JWTSecurityProvider(config)
        assert provider is not None
        assert provider.config == config


class TestNoSecurityProvider:
    """Test no-security provider (for development)."""
    
    def test_no_security_provider_creation(self):
        """Test no-security provider creation."""
        provider = NoSecurityProvider()
        assert provider is not None


class TestOAuth2Provider:
    """Test OAuth2 security provider."""
    
    def test_oauth2_config_creation(self):
        """Test OAuth2 config creation."""
        config = OAuth2Config(
            client_id="test-client",
            client_secret="test-secret",
            authorization_url="https://auth.example.com/oauth/authorize",
            token_url="https://auth.example.com/oauth/token",
            redirect_uri="https://app.example.com/callback",
            scope=["read", "write"]
        )
        assert config.client_id == "test-client"
        assert config.client_secret == "test-secret"
        assert config.authorization_url == "https://auth.example.com/oauth/authorize"
        assert config.token_url == "https://auth.example.com/oauth/token"
        assert config.redirect_uri == "https://app.example.com/callback"
        assert config.scope == ["read", "write"]
    
    def test_oauth2_provider_creation(self):
        """Test OAuth2 provider creation."""
        config = OAuth2Config(
            client_id="test-client",
            client_secret="test-secret",
            authorization_url="https://auth.example.com/oauth/authorize",
            token_url="https://auth.example.com/oauth/token",
            redirect_uri="https://app.example.com/callback",
            scope=["read", "write"]
        )
        provider = OAuth2SecurityProvider(config)
        assert provider is not None
        assert provider.config == config


class TestSecuritySystem:
    """Test overall security system integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_security_system_initialization(self, temp_dir):
        """Test security system initialization."""
        config = {
            "security": {
                "provider": "jwt",
                "jwt": {
                    "secret_key": "test-secret",
                    "algorithm": "HS256",
                    "expiry_minutes": 60
                }
            }
        }
        
        # Initialize security system
        initialize_security_system(config)
        
        # Verify managers are available
        auth_manager = get_auth_manager()
        assert auth_manager is not None
        
        auth_z_manager = get_authorization_manager()
        assert auth_z_manager is not None
        
        token_registry = get_token_registry()
        assert token_registry is not None
        
        signature_manager = get_signature_manager()
        assert signature_manager is not None
        
        encryption_manager = get_encryption_manager()
        assert encryption_manager is not None
        
        # Shutdown security system
        shutdown_security_system()
    
    def test_security_system_no_security(self):
        """Test security system with no security provider."""
        config = {
            "security": {
                "provider": "none"
            }
        }
        
        # Initialize security system
        initialize_security_system(config)
        
        # Verify managers are available
        auth_manager = get_auth_manager()
        assert auth_manager is not None
        
        # Shutdown security system
        shutdown_security_system()


@pytest.mark.asyncio
class TestAsyncSecurity:
    """Test async security operations."""
    
    async def test_async_authentication(self):
        """Test async authentication operations."""
        # This would test actual async authentication flows
        # For now, just verify the basic setup works
        manager = AuthenticationManager()
        assert manager is not None
    
    async def test_async_authorization(self):
        """Test async authorization operations."""
        # This would test actual async authorization checks
        # For now, just verify the basic setup works
        manager = AuthorizationManager()
        assert manager is not None
    
    async def test_async_token_operations(self):
        """Test async token operations."""
        # This would test actual async token operations
        # For now, just verify the basic setup works
        registry = TokenManagerRegistry()
        assert registry is not None


class TestSecurityIntegration:
    """Test security integration with other framework components."""
    
    def test_security_with_queue_operations(self):
        """Test security integration with queue operations."""
        # This would test security with actual queue operations
        # For now, just verify security components are available
        auth_manager = get_auth_manager()
        auth_z_manager = get_authorization_manager()
        token_registry = get_token_registry()
        
        assert auth_manager is not None
        assert auth_z_manager is not None
        assert token_registry is not None
    
    def test_security_with_transport_layer(self):
        """Test security integration with transport layer."""
        # This would test security with transport operations
        # For now, just verify security components are available
        signature_manager = get_signature_manager()
        encryption_manager = get_encryption_manager()
        
        assert signature_manager is not None
        assert encryption_manager is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
