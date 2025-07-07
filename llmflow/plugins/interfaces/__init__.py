"""
LLMFlow Plugin Interfaces

This module provides the core plugin interfaces that define the contract for
all pluggable components in the LLMFlow framework.
"""

from .base import Plugin, PluginError, PluginInitializationError
from .transport import ITransportProtocol, TransportError
from .security import ISecurityProvider, SecurityError
from .serialization import IMessageSerializer, SerializationError
from .storage import IStorageProvider, StorageError
from .monitoring import IMonitoringProvider, MonitoringError

__all__ = [
    # Base plugin interface
    'Plugin',
    'PluginError',
    'PluginInitializationError',
    
    # Transport interface
    'ITransportProtocol',
    'TransportError',
    
    # Security interface
    'ISecurityProvider',
    'SecurityError',
    
    # Serialization interface
    'IMessageSerializer',
    'SerializationError',
    
    # Storage interface
    'IStorageProvider',
    'StorageError',
    
    # Monitoring interface
    'IMonitoringProvider',
    'MonitoringError',
]
