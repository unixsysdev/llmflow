"""
OAuth2 Security Provider

This module provides OAuth2-based authentication for LLMFlow.
"""

import logging
import httpx
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..interfaces.security import ISecurityProvider, SecurityLevel, Token, SecurityContext
from ..interfaces.security import AuthenticationError, AuthorizationError, SecurityError
from ..interfaces.base import Plugin

logger = logging.getLogger(__name__)


class OAuth2GrantType(Enum):
    """OAuth2 grant types."""
    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"


@dataclass
class OAuth2Config:
    """Configuration for OAuth2 provider."""
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    userinfo_url: str
    redirect_uri: str
    scopes: List[str] = None
    timeout: int = 30
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = ["openid", "profile", "email"]


class OAuth2SecurityProvider(Plugin, ISecurityProvider):
    """OAuth2-based security provider."""
    
    def __init__(self, config: Dict[str, Any] = None):
        Plugin.__init__(self, config)
        self.oauth2_config = OAuth2Config(**self.config)
        self._http_client: Optional[httpx.AsyncClient] = None
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate OAuth2 configuration."""
        required_fields = ["client_id", "client_secret", "authorization_url", "token_url", "userinfo_url"]
        for field in required_fields:
            if not getattr(self.oauth2_config, field):
                raise SecurityError(f"OAuth2 {field} is required")
    
    # Plugin interface methods
    def get_name(self) -> str:
        return "oauth2_security_provider"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "OAuth2-based authentication provider"
    
    def get_dependencies(self) -> List[str]:
        return ["httpx"]
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the OAuth2 provider."""
        self.config.update(config)
        self.oauth2_config = OAuth2Config(**self.config)
        self._validate_config()
        
        # Initialize HTTP client
        self._http_client = httpx.AsyncClient(timeout=self.oauth2_config.timeout)
        
        logger.info("OAuth2 security provider initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the OAuth2 provider."""
        if self._http_client:
            await self._http_client.aclose()
        logger.info("OAuth2 security provider shutdown")
    
    # Security provider interface methods
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Token]:
        """Authenticate using OAuth2 flow."""
        try:
            grant_type = credentials.get("grant_type", OAuth2GrantType.AUTHORIZATION_CODE.value)
            
            if grant_type == OAuth2GrantType.AUTHORIZATION_CODE.value:
                return await self._authenticate_authorization_code(credentials)
            elif grant_type == OAuth2GrantType.CLIENT_CREDENTIALS.value:
                return await self._authenticate_client_credentials(credentials)
            elif grant_type == OAuth2GrantType.REFRESH_TOKEN.value:
                return await self._refresh_token(credentials)
            else:
                raise AuthenticationError(f"Unsupported grant type: {grant_type}")
                
        except Exception as e:
            logger.error(f"OAuth2 authentication failed: {e}")
            if isinstance(e, AuthenticationError):
                raise
            raise AuthenticationError(f"OAuth2 authentication failed: {e}")
    
    async def authorize(self, token: Token, resource: str, action: str) -> bool:
        """Authorize access using OAuth2 token."""
        try:
            # Get user info from OAuth2 provider
            user_info = await self._get_user_info(token)
            
            # Check scopes/permissions
            user_scopes = user_info.get("scopes", [])
            required_scope = f"{resource}:{action}"
            
            # Simple scope-based authorization
            authorized = (
                required_scope in user_scopes or
                f"{resource}:*" in user_scopes or
                "admin" in user_scopes
            )
            
            if authorized:
                logger.debug(f"User {user_info.get('sub')} authorized for {required_scope}")
            else:
                logger.warning(f"User {user_info.get('sub')} denied access to {required_scope}")
            
            return authorized
            
        except Exception as e:
            logger.error(f"OAuth2 authorization failed: {e}")
            raise AuthorizationError(f"OAuth2 authorization failed: {e}")
    
    async def encrypt(self, data: bytes, key: str) -> bytes:
        """Encrypt data (not implemented in OAuth2 provider)."""
        raise NotImplementedError("Encryption not supported by OAuth2 provider")
    
    async def decrypt(self, data: bytes, key: str) -> bytes:
        """Decrypt data (not implemented in OAuth2 provider)."""
        raise NotImplementedError("Decryption not supported by OAuth2 provider")
    
    async def sign(self, data: bytes, key: str) -> bytes:
        """Sign data (not implemented in OAuth2 provider)."""
        raise NotImplementedError("Signing not supported by OAuth2 provider")
    
    async def verify(self, data: bytes, signature: bytes, key: str) -> bool:
        """Verify signature (not implemented in OAuth2 provider)."""
        raise NotImplementedError("Signature verification not supported by OAuth2 provider")
    
    async def generate_key(self) -> str:
        """Generate OAuth2 state parameter."""
        import secrets
        return secrets.token_urlsafe(32)
    
    async def create_security_context(self, token: Token) -> SecurityContext:
        """Create security context from OAuth2 token."""
        try:
            user_info = await self._get_user_info(token)
            
            return SecurityContext(
                user_id=user_info.get("sub"),
                permissions=user_info.get("scopes", []),
                security_level=SecurityLevel.AUTHENTICATED,
                domain=user_info.get("domain", "default"),
                tenant_id=user_info.get("tenant_id", "default"),
                expires_at=token.expires_at
            )
            
        except Exception as e:
            raise SecurityError(f"Failed to create security context: {e}")
    
    def get_authorization_url(self, state: str = None) -> str:
        """Get OAuth2 authorization URL."""
        if state is None:
            import secrets
            state = secrets.token_urlsafe(32)
        
        params = {
            "response_type": "code",
            "client_id": self.oauth2_config.client_id,
            "redirect_uri": self.oauth2_config.redirect_uri,
            "scope": " ".join(self.oauth2_config.scopes),
            "state": state
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.oauth2_config.authorization_url}?{query_string}"
    
    # Internal helper methods
    async def _authenticate_authorization_code(self, credentials: Dict[str, Any]) -> Token:
        """Authenticate using authorization code grant."""
        code = credentials.get("code")
        state = credentials.get("state")
        
        if not code:
            raise AuthenticationError("Authorization code is required")
        
        # Exchange code for token
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.oauth2_config.redirect_uri,
            "client_id": self.oauth2_config.client_id,
            "client_secret": self.oauth2_config.client_secret
        }
        
        response = await self._http_client.post(
            self.oauth2_config.token_url,
            data=token_data
        )
        
        if response.status_code != 200:
            raise AuthenticationError(f"Token exchange failed: {response.text}")
        
        token_response = response.json()
        return self._create_token_from_response(token_response)
    
    async def _authenticate_client_credentials(self, credentials: Dict[str, Any]) -> Token:
        """Authenticate using client credentials grant."""
        token_data = {
            "grant_type": "client_credentials",
            "client_id": self.oauth2_config.client_id,
            "client_secret": self.oauth2_config.client_secret,
            "scope": " ".join(self.oauth2_config.scopes)
        }
        
        response = await self._http_client.post(
            self.oauth2_config.token_url,
            data=token_data
        )
        
        if response.status_code != 200:
            raise AuthenticationError(f"Client credentials authentication failed: {response.text}")
        
        token_response = response.json()
        return self._create_token_from_response(token_response)
    
    async def _refresh_token(self, credentials: Dict[str, Any]) -> Token:
        """Refresh OAuth2 token."""
        refresh_token = credentials.get("refresh_token")
        
        if not refresh_token:
            raise AuthenticationError("Refresh token is required")
        
        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.oauth2_config.client_id,
            "client_secret": self.oauth2_config.client_secret
        }
        
        response = await self._http_client.post(
            self.oauth2_config.token_url,
            data=token_data
        )
        
        if response.status_code != 200:
            raise AuthenticationError(f"Token refresh failed: {response.text}")
        
        token_response = response.json()
        return self._create_token_from_response(token_response)
    
    async def _get_user_info(self, token: Token) -> Dict[str, Any]:
        """Get user information from OAuth2 provider."""
        headers = {"Authorization": f"Bearer {token.token}"}
        
        response = await self._http_client.get(
            self.oauth2_config.userinfo_url,
            headers=headers
        )
        
        if response.status_code != 200:
            raise AuthenticationError(f"Failed to get user info: {response.text}")
        
        return response.json()
    
    def _create_token_from_response(self, token_response: Dict[str, Any]) -> Token:
        """Create Token object from OAuth2 response."""
        access_token = token_response.get("access_token")
        token_type = token_response.get("token_type", "bearer")
        expires_in = token_response.get("expires_in", 3600)
        
        if not access_token:
            raise AuthenticationError("No access token in response")
        
        expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        return Token(
            token=access_token,
            token_type=token_type,
            expires_at=expires_at,
            user_id=None,  # Will be filled when user info is retrieved
            refresh_token=token_response.get("refresh_token")
        )
