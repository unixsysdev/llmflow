"""
LLMFlow Security Module

This module provides comprehensive security functionality for LLMFlow,
including authentication, authorization, encryption, and message signing.
"""

# Import all security providers
from .providers import (
    JWTSecurityProvider, JWTConfig,
    OAuth2SecurityProvider, OAuth2Config,
    NoSecurityProvider
)

# Import authentication and authorization
from .auth import (
    Authenticator, AuthenticationManager, get_auth_manager,
    Authorizer, AuthorizationManager, Permission, Role, get_authorization_manager,
    TokenManager, TokenManagerRegistry, TokenInfo, get_token_registry
)

# Import cryptography
from .crypto import (
    MessageSigner, SignatureManager, get_signature_manager,
    MessageEncryptor, EncryptionManager, SecureMessageEnvelope, get_encryption_manager
)

# Import interfaces
from .interfaces.security import (
    ISecurityProvider, SecurityLevel, AuthenticationMethod,
    Token, SecurityContext,
    SecurityError, AuthenticationError, AuthorizationError, CryptographyError
)

__all__ = [
    # Security Providers
    'JWTSecurityProvider',
    'JWTConfig',
    'OAuth2SecurityProvider', 
    'OAuth2Config',
    'NoSecurityProvider',
    
    # Authentication & Authorization
    'Authenticator',
    'AuthenticationManager',
    'get_auth_manager',
    'Authorizer',
    'AuthorizationManager', 
    'Permission',
    'Role',
    'get_authorization_manager',
    'TokenManager',
    'TokenManagerRegistry',
    'TokenInfo',
    'get_token_registry',
    
    # Cryptography
    'MessageSigner',
    'SignatureManager',
    'get_signature_manager',
    'MessageEncryptor',
    'EncryptionManager', 
    'SecureMessageEnvelope',
    'get_encryption_manager',
    
    # Interfaces & Types
    'ISecurityProvider',
    'SecurityLevel',
    'AuthenticationMethod',
    'Token',
    'SecurityContext',
    'SecurityError',
    'AuthenticationError',
    'AuthorizationError',
    'CryptographyError'
]


# Security system initialization
async def initialize_security_system(config: dict = None) -> None:
    """Initialize the LLMFlow security system."""
    from logging import getLogger
    logger = getLogger(__name__)
    
    config = config or {}
    
    # Initialize default security providers
    if config.get("jwt_enabled", True):
        jwt_config = config.get("jwt", {
            "secret_key": "development_secret_key_change_in_production",
            "access_token_expire_minutes": 30
        })
        
        jwt_provider = JWTSecurityProvider(jwt_config)
        
        # Register with authentication manager
        auth_manager = get_auth_manager()
        auth_manager.register_provider("jwt", jwt_provider, is_default=True)
        
        # Register with authorization manager  
        authz_manager = get_authorization_manager()
        authz_manager.register_provider("jwt", jwt_provider, is_default=True)
        
        # Register with token manager
        token_registry = get_token_registry()
        token_registry.register_provider("jwt", jwt_provider, is_default=True)
        
        logger.info("JWT security provider initialized")
    
    # Initialize development no-security provider if enabled
    if config.get("development_mode", False):
        no_security_provider = NoSecurityProvider({"development_mode": True})
        
        auth_manager = get_auth_manager()
        auth_manager.register_provider("no_security", no_security_provider)
        
        logger.warning("⚠️  Development no-security provider enabled!")
    
    # Initialize cryptography managers
    signature_manager = get_signature_manager()
    encryption_manager = get_encryption_manager()
    
    # Generate default keys if not provided
    if config.get("auto_generate_keys", True):
        await signature_manager.generate_and_add_key("default", is_default=True)
        await encryption_manager.add_symmetric_key("default", is_default=True)
        logger.info("Default cryptographic keys generated")
    
    # Start token managers
    token_registry = get_token_registry()
    await token_registry.start_all()
    
    logger.info("LLMFlow security system initialized")


async def shutdown_security_system() -> None:
    """Shutdown the LLMFlow security system."""
    from logging import getLogger
    logger = getLogger(__name__)
    
    # Stop token managers
    token_registry = get_token_registry()
    await token_registry.stop_all()
    
    logger.info("LLMFlow security system shutdown")
