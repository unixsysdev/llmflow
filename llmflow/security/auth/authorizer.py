"""
Authorization Module

This module provides authorization logic for LLMFlow.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from datetime import datetime

from ..interfaces.security import ISecurityProvider, Token, SecurityContext
from ..interfaces.security import AuthorizationError, SecurityError

logger = logging.getLogger(__name__)


class Permission:
    """Represents a permission with resource and action."""
    
    def __init__(self, resource: str, action: str):
        self.resource = resource
        self.action = action
    
    def __str__(self) -> str:
        return f"{self.resource}:{self.action}"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Permission):
            return self.resource == other.resource and self.action == other.action
        return False
    
    def __hash__(self) -> bool:
        return hash((self.resource, self.action))
    
    def matches(self, other: 'Permission') -> bool:
        """Check if this permission matches another (supports wildcards)."""
        if self.resource == "*" or other.resource == "*":
            return True
        if self.action == "*" or other.action == "*":
            return self.resource == other.resource
        return self == other


class Role:
    """Represents a role with permissions."""
    
    def __init__(self, name: str, permissions: List[Permission] = None):
        self.name = name
        self.permissions = set(permissions or [])
    
    def add_permission(self, permission: Permission) -> None:
        """Add a permission to this role."""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission) -> None:
        """Remove a permission from this role."""
        self.permissions.discard(permission)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has a specific permission."""
        for role_permission in self.permissions:
            if role_permission.matches(permission):
                return True
        return False


class Authorizer:
    """Handles authorization operations."""
    
    def __init__(self, security_provider: ISecurityProvider):
        self.security_provider = security_provider
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self._setup_default_roles()
    
    def _setup_default_roles(self) -> None:
        """Setup default roles."""
        # Admin role - all permissions
        admin_role = Role("admin", [Permission("*", "*")])
        self.roles["admin"] = admin_role
        
        # User role - basic permissions
        user_permissions = [
            Permission("queue", "read"),
            Permission("queue", "write"),
            Permission("transport", "send"),
            Permission("transport", "receive")
        ]
        user_role = Role("user", user_permissions)
        self.roles["user"] = user_role
        
        # Guest role - read-only permissions
        guest_permissions = [
            Permission("queue", "read"),
            Permission("transport", "receive")
        ]
        guest_role = Role("guest", guest_permissions)
        self.roles["guest"] = guest_role
    
    async def authorize(self, token: Token, resource: str, action: str) -> bool:
        """Authorize access to a resource."""
        try:
            # First check with security provider
            provider_result = await self.security_provider.authorize(token, resource, action)
            
            if provider_result:
                return True
            
            # Then check with local RBAC
            security_context = await self.security_provider.create_security_context(token)
            return await self.authorize_context(security_context, resource, action)
            
        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            raise AuthorizationError(f"Authorization failed: {e}")
    
    async def authorize_context(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Authorize using security context."""
        try:
            permission = Permission(resource, action)
            
            # Check context permissions
            for context_permission in context.permissions:
                if ":" in context_permission:
                    res, act = context_permission.split(":", 1)
                    if Permission(res, act).matches(permission):
                        logger.debug(f"Authorized {context.user_id} for {permission} via context")
                        return True
            
            # Check user roles
            user_roles = self.user_roles.get(context.user_id, set())
            for role_name in user_roles:
                role = self.roles.get(role_name)
                if role and role.has_permission(permission):
                    logger.debug(f"Authorized {context.user_id} for {permission} via role {role_name}")
                    return True
            
            logger.warning(f"Access denied for {context.user_id} to {permission}")
            return False
            
        except Exception as e:
            logger.error(f"Context authorization failed: {e}")
            return False
    
    def add_role(self, role: Role) -> None:
        """Add a role to the system."""
        self.roles[role.name] = role
        logger.info(f"Added role: {role.name}")
    
    def remove_role(self, role_name: str) -> bool:
        """Remove a role from the system."""
        if role_name in self.roles:
            del self.roles[role_name]
            # Remove role from all users
            for user_id in self.user_roles:
                self.user_roles[user_id].discard(role_name)
            logger.info(f"Removed role: {role_name}")
            return True
        return False
    
    def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        if role_name not in self.roles:
            logger.error(f"Role {role_name} does not exist")
            return False
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role_name)
        logger.info(f"Assigned role {role_name} to user {user_id}")
        return True
    
    def remove_role_from_user(self, user_id: str, role_name: str) -> bool:
        """Remove a role from a user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
            logger.info(f"Removed role {role_name} from user {user_id}")
            return True
        return False
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """Get roles assigned to a user."""
        return self.user_roles.get(user_id, set())
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user."""
        permissions = set()
        user_roles = self.get_user_roles(user_id)
        
        for role_name in user_roles:
            role = self.roles.get(role_name)
            if role:
                permissions.update(role.permissions)
        
        return permissions
    
    def list_roles(self) -> List[str]:
        """List all available roles."""
        return list(self.roles.keys())
    
    def get_role(self, role_name: str) -> Optional[Role]:
        """Get a role by name."""
        return self.roles.get(role_name)


class AuthorizationManager:
    """High-level authorization manager."""
    
    def __init__(self):
        self.authorizers: Dict[str, Authorizer] = {}
        self.default_provider: Optional[str] = None
    
    def register_provider(self, name: str, security_provider: ISecurityProvider, is_default: bool = False) -> None:
        """Register a security provider."""
        self.authorizers[name] = Authorizer(security_provider)
        
        if is_default or self.default_provider is None:
            self.default_provider = name
        
        logger.info(f"Registered authorization provider: {name}")
    
    def get_authorizer(self, provider_name: str = None) -> Authorizer:
        """Get authorizer for a provider."""
        name = provider_name or self.default_provider
        
        if name not in self.authorizers:
            raise SecurityError(f"Authorization provider '{name}' not found")
        
        return self.authorizers[name]
    
    async def authorize(self, token: Token, resource: str, action: str, provider_name: str = None) -> bool:
        """Authorize using specified or default provider."""
        authorizer = self.get_authorizer(provider_name)
        return await authorizer.authorize(token, resource, action)
    
    def assign_role_to_user(self, user_id: str, role_name: str, provider_name: str = None) -> bool:
        """Assign role to user."""
        authorizer = self.get_authorizer(provider_name)
        return authorizer.assign_role_to_user(user_id, role_name)


# Global authorization manager instance
_auth_manager = AuthorizationManager()

def get_authorization_manager() -> AuthorizationManager:
    """Get the global authorization manager."""
    return _auth_manager
