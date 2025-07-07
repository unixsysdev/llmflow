"""
LLMFlow Authentication Molecules

This module contains molecules for authentication and authorization functionality.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import hashlib
import hmac
import base64
import logging

from ..core.base import DataAtom, ServiceAtom, ValidationResult
from ..atoms.data import StringAtom, BooleanAtom, TimestampAtom, UUIDAtom
from ..atoms.service import (
    ValidateEmailAtom, HashPasswordAtom, VerifyPasswordAtom,
    GenerateTokenAtom, VerifyTokenAtom, GenerateUUIDAtom
)
from ..queue import QueueClient, QueueManager

logger = logging.getLogger(__name__)


class UserCredentialsAtom(DataAtom):
    """Data atom for user credentials."""
    
    def __init__(self, email: str, password: str, metadata: Dict[str, Any] = None):
        super().__init__({'email': email, 'password': password}, metadata)
        self.email = email
        self.password = password
    
    def validate(self) -> ValidationResult:
        """Validate user credentials."""
        if not self.email or not self.password:
            return ValidationResult.error("Email and password are required")
        
        if '@' not in self.email:
            return ValidationResult.error("Invalid email format")
        
        if len(self.password) < 8:
            return ValidationResult.error("Password must be at least 8 characters")
        
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize credentials to bytes."""
        import msgpack
        return msgpack.packb(self.value)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'UserCredentialsAtom':
        """Deserialize bytes to credentials."""
        import msgpack
        value = msgpack.unpackb(data, raw=False)
        return cls(value['email'], value['password'])


class AuthTokenAtom(DataAtom):
    """Data atom for authentication tokens."""
    
    def __init__(self, token: str, user_id: str, expires_at: datetime, 
                 metadata: Dict[str, Any] = None):
        super().__init__({
            'token': token,
            'user_id': user_id,
            'expires_at': expires_at,
            'issued_at': datetime.utcnow()
        }, metadata)
        self.token = token
        self.user_id = user_id
        self.expires_at = expires_at
    
    def validate(self) -> ValidationResult:
        """Validate authentication token."""
        if not self.token:
            return ValidationResult.error("Token is required")
        
        if not self.user_id:
            return ValidationResult.error("User ID is required")
        
        if self.expires_at <= datetime.utcnow():
            return ValidationResult.error("Token has expired")
        
        return ValidationResult.success()
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return self.expires_at <= datetime.utcnow()
    
    def serialize(self) -> bytes:
        """Serialize token to bytes."""
        import msgpack
        data = self.value.copy()
        data['expires_at'] = data['expires_at'].isoformat()
        data['issued_at'] = data['issued_at'].isoformat()
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'AuthTokenAtom':
        """Deserialize bytes to token."""
        import msgpack
        value = msgpack.unpackb(data, raw=False)
        expires_at = datetime.fromisoformat(value['expires_at'])
        return cls(value['token'], value['user_id'], expires_at)


class UserSessionAtom(DataAtom):
    """Data atom for user sessions."""
    
    def __init__(self, session_id: str, user_id: str, ip_address: str,
                 user_agent: str, metadata: Dict[str, Any] = None):
        super().__init__({
            'session_id': session_id,
            'user_id': user_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow()
        }, metadata)
        self.session_id = session_id
        self.user_id = user_id
        self.ip_address = ip_address
        self.user_agent = user_agent
    
    def validate(self) -> ValidationResult:
        """Validate user session."""
        if not self.session_id:
            return ValidationResult.error("Session ID is required")
        
        if not self.user_id:
            return ValidationResult.error("User ID is required")
        
        return ValidationResult.success()
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.value['last_activity'] = datetime.utcnow()
    
    def serialize(self) -> bytes:
        """Serialize session to bytes."""
        import msgpack
        data = self.value.copy()
        data['created_at'] = data['created_at'].isoformat()
        data['last_activity'] = data['last_activity'].isoformat()
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'UserSessionAtom':
        """Deserialize bytes to session."""
        import msgpack
        value = msgpack.unpackb(data, raw=False)
        return cls(
            value['session_id'],
            value['user_id'],
            value['ip_address'],
            value['user_agent']
        )


class AuthenticationMolecule(ServiceAtom):
    """Service molecule for user authentication."""
    
    def __init__(self, queue_manager: QueueManager, secret_key: str):
        super().__init__(
            name="authentication_molecule",
            input_types=[
                "llmflow.molecules.auth.UserCredentialsAtom"
            ],
            output_types=[
                "llmflow.molecules.auth.AuthTokenAtom",
                "llmflow.atoms.data.BooleanAtom"
            ]
        )
        self.queue_manager = queue_manager
        self.secret_key = secret_key
        self.user_store = {}  # In production, this would be a database
        
        # Initialize service atoms
        self.validate_email = ValidateEmailAtom()
        self.hash_password = HashPasswordAtom()
        self.verify_password = VerifyPasswordAtom()
        self.generate_token = GenerateTokenAtom()
        self.generate_uuid = GenerateUUIDAtom()
    
    async def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Process authentication request."""
        credentials = inputs[0]
        
        # Validate input
        if not isinstance(credentials, UserCredentialsAtom):
            return [BooleanAtom(False)]
        
        validation_result = credentials.validate()
        if not validation_result.is_valid:
            logger.warning(f"Invalid credentials: {validation_result.errors}")
            return [BooleanAtom(False)]
        
        # Validate email format
        email_valid = self.validate_email.process([StringAtom(credentials.email)])[0]
        if not email_valid.value:
            logger.warning(f"Invalid email format: {credentials.email}")
            return [BooleanAtom(False)]
        
        # Check if user exists and password is correct
        user_id = await self._authenticate_user(credentials.email, credentials.password)
        if not user_id:
            logger.warning(f"Authentication failed for: {credentials.email}")
            return [BooleanAtom(False)]
        
        # Generate authentication token
        token = await self._generate_auth_token(user_id)
        
        logger.info(f"Authentication successful for user: {user_id}")
        return [token, BooleanAtom(True)]
    
    async def _authenticate_user(self, email: str, password: str) -> Optional[str]:
        """Authenticate user credentials."""
        # In production, this would query a database
        # For now, we'll use a simple in-memory store
        user_data = self.user_store.get(email)
        
        if not user_data:
            # User doesn't exist, create them for demo purposes
            user_id = str(uuid.uuid4())
            hashed_password = self.hash_password.process([StringAtom(password)])[0]
            
            self.user_store[email] = {
                'user_id': user_id,
                'email': email,
                'password_hash': hashed_password.value,
                'created_at': datetime.utcnow()
            }
            
            return user_id
        
        # Verify password
        password_valid = self.verify_password.process([
            StringAtom(password),
            StringAtom(user_data['password_hash'])
        ])[0]
        
        if password_valid.value:
            return user_data['user_id']
        
        return None
    
    async def _generate_auth_token(self, user_id: str) -> AuthTokenAtom:
        """Generate authentication token."""
        # Generate token
        token_data = {
            'user_id': user_id,
            'issued_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }
        
        token = self.generate_token.process([StringAtom(str(token_data))])[0]
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        return AuthTokenAtom(token.value, user_id, expires_at)


class SessionMolecule(ServiceAtom):
    """Service molecule for session management."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="session_molecule",
            input_types=[
                "llmflow.molecules.auth.AuthTokenAtom",
                "llmflow.atoms.data.StringAtom",  # IP address
                "llmflow.atoms.data.StringAtom"   # User agent
            ],
            output_types=[
                "llmflow.molecules.auth.UserSessionAtom",
                "llmflow.atoms.data.BooleanAtom"
            ]
        )
        self.queue_manager = queue_manager
        self.session_store = {}  # In production, this would be a database
        self.generate_uuid = GenerateUUIDAtom()
    
    async def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Process session creation request."""
        auth_token = inputs[0]
        ip_address = inputs[1]
        user_agent = inputs[2]
        
        # Validate token
        if not isinstance(auth_token, AuthTokenAtom):
            return [BooleanAtom(False)]
        
        validation_result = auth_token.validate()
        if not validation_result.is_valid:
            logger.warning(f"Invalid token: {validation_result.errors}")
            return [BooleanAtom(False)]
        
        if auth_token.is_expired():
            logger.warning("Token has expired")
            return [BooleanAtom(False)]
        
        # Create session
        session_id = self.generate_uuid.process([])[0].value
        session = UserSessionAtom(
            session_id=session_id,
            user_id=auth_token.user_id,
            ip_address=ip_address.value,
            user_agent=user_agent.value
        )
        
        # Store session
        self.session_store[session_id] = session
        
        logger.info(f"Session created for user: {auth_token.user_id}")
        return [session, BooleanAtom(True)]
    
    async def validate_session(self, session_id: str) -> Optional[UserSessionAtom]:
        """Validate and retrieve session."""
        session = self.session_store.get(session_id)
        
        if session:
            # Update last activity
            session.update_activity()
            return session
        
        return None
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        if session_id in self.session_store:
            del self.session_store[session_id]
            logger.info(f"Session invalidated: {session_id}")
            return True
        
        return False


class AuthorizationMolecule(ServiceAtom):
    """Service molecule for authorization and access control."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="authorization_molecule",
            input_types=[
                "llmflow.molecules.auth.UserSessionAtom",
                "llmflow.atoms.data.StringAtom",  # Resource
                "llmflow.atoms.data.StringAtom"   # Action
            ],
            output_types=[
                "llmflow.atoms.data.BooleanAtom"
            ]
        )
        self.queue_manager = queue_manager
        self.permissions_store = {}  # In production, this would be a database
        
        # Initialize default permissions
        self._initialize_permissions()
    
    def _initialize_permissions(self) -> None:
        """Initialize default permissions."""
        # Default permissions for demonstration
        self.permissions_store = {
            'default': {
                'queue.read': True,
                'queue.write': True,
                'system.health': False,
                'system.metrics': False
            },
            'admin': {
                'queue.read': True,
                'queue.write': True,
                'queue.create': True,
                'queue.delete': True,
                'system.health': True,
                'system.metrics': True,
                'system.admin': True
            }
        }
    
    async def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Process authorization request."""
        session = inputs[0]
        resource = inputs[1]
        action = inputs[2]
        
        # Validate inputs
        if not isinstance(session, UserSessionAtom):
            return [BooleanAtom(False)]
        
        validation_result = session.validate()
        if not validation_result.is_valid:
            logger.warning(f"Invalid session: {validation_result.errors}")
            return [BooleanAtom(False)]
        
        # Check authorization
        is_authorized = await self._check_authorization(
            session.user_id,
            resource.value,
            action.value
        )
        
        if is_authorized:
            logger.debug(f"Authorized: {session.user_id} -> {resource.value}:{action.value}")
        else:
            logger.warning(f"Unauthorized: {session.user_id} -> {resource.value}:{action.value}")
        
        return [BooleanAtom(is_authorized)]
    
    async def _check_authorization(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user is authorized for resource and action."""
        # Get user role (simplified - in production, this would be from database)
        user_role = self._get_user_role(user_id)
        
        # Check permissions
        permissions = self.permissions_store.get(user_role, {})
        permission_key = f"{resource}.{action}"
        
        return permissions.get(permission_key, False)
    
    def _get_user_role(self, user_id: str) -> str:
        """Get user role (simplified implementation)."""
        # In production, this would query a database
        # For now, return 'default' for all users
        return 'default'
    
    async def grant_permission(self, user_role: str, resource: str, action: str) -> bool:
        """Grant permission to a role."""
        if user_role not in self.permissions_store:
            self.permissions_store[user_role] = {}
        
        permission_key = f"{resource}.{action}"
        self.permissions_store[user_role][permission_key] = True
        
        logger.info(f"Granted permission: {user_role} -> {permission_key}")
        return True
    
    async def revoke_permission(self, user_role: str, resource: str, action: str) -> bool:
        """Revoke permission from a role."""
        if user_role in self.permissions_store:
            permission_key = f"{resource}.{action}"
            self.permissions_store[user_role][permission_key] = False
            
            logger.info(f"Revoked permission: {user_role} -> {permission_key}")
            return True
        
        return False


class AuthFlowMolecule(ServiceAtom):
    """Complete authentication flow molecule."""
    
    def __init__(self, queue_manager: QueueManager, secret_key: str):
        super().__init__(
            name="auth_flow_molecule",
            input_types=[
                "llmflow.molecules.auth.UserCredentialsAtom",
                "llmflow.atoms.data.StringAtom",  # IP address
                "llmflow.atoms.data.StringAtom"   # User agent
            ],
            output_types=[
                "llmflow.molecules.auth.UserSessionAtom",
                "llmflow.atoms.data.BooleanAtom"
            ]
        )
        self.queue_manager = queue_manager
        
        # Initialize sub-molecules
        self.auth_molecule = AuthenticationMolecule(queue_manager, secret_key)
        self.session_molecule = SessionMolecule(queue_manager)
    
    async def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Process complete authentication flow."""
        credentials = inputs[0]
        ip_address = inputs[1]
        user_agent = inputs[2]
        
        # Step 1: Authenticate user
        auth_result = await self.auth_molecule.process([credentials])
        
        if len(auth_result) < 2 or not auth_result[1].value:
            logger.warning("Authentication failed")
            return [BooleanAtom(False)]
        
        auth_token = auth_result[0]
        
        # Step 2: Create session
        session_result = await self.session_molecule.process([
            auth_token,
            ip_address,
            user_agent
        ])
        
        if len(session_result) < 2 or not session_result[1].value:
            logger.warning("Session creation failed")
            return [BooleanAtom(False)]
        
        session = session_result[0]
        
        logger.info(f"Complete auth flow successful for session: {session.session_id}")
        return [session, BooleanAtom(True)]
