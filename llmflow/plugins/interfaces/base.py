"""
Base Plugin Interface

This module defines the core plugin interface that all LLMFlow plugins must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Plugin lifecycle status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginInitializationError(PluginError):
    """Raised when plugin initialization fails."""
    pass


class PluginValidationError(PluginError):
    """Raised when plugin validation fails."""
    pass


class Plugin(ABC):
    """
    Base interface for all LLMFlow plugins.
    
    This interface defines the contract that all plugins must implement to be
    compatible with the LLMFlow framework's plugin system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the plugin.
        
        Args:
            config: Plugin-specific configuration dictionary
        """
        self.config = config or {}
        self.status = PluginStatus.UNINITIALIZED
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the plugin name.
        
        Returns:
            The unique name of the plugin
        """
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """
        Get the plugin version.
        
        Returns:
            The version string of the plugin
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get the plugin description.
        
        Returns:
            A human-readable description of the plugin
        """
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Get the list of plugin dependencies.
        
        Returns:
            List of plugin names that this plugin depends on
        """
        pass
    
    @abstractmethod
    def get_interfaces(self) -> List[Type]:
        """
        Get the list of interfaces this plugin implements.
        
        Returns:
            List of interface types that this plugin implements
        """
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with the given configuration.
        
        Args:
            config: Plugin-specific configuration
            
        Raises:
            PluginInitializationError: If initialization fails
        """
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start the plugin.
        
        This method is called after initialization to start the plugin's
        background processes, if any.
        
        Raises:
            PluginError: If starting fails
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the plugin.
        
        This method is called to cleanly shut down the plugin and release
        all resources.
        
        Raises:
            PluginError: If stopping fails
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the plugin completely.
        
        This method is called for final cleanup and should release all
        resources held by the plugin.
        
        Raises:
            PluginError: If shutdown fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the plugin is healthy and operational.
        
        Returns:
            True if the plugin is healthy, False otherwise
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current plugin configuration.
        
        Returns:
            The current configuration dictionary
        """
        return self.config.copy()
    
    def get_status(self) -> PluginStatus:
        """
        Get the current plugin status.
        
        Returns:
            The current plugin status
        """
        return self.status
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Raises:
            PluginValidationError: If configuration is invalid
        """
        # Default implementation - subclasses can override
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.
        
        Returns:
            Dictionary containing plugin metadata
        """
        return {
            'name': self.get_name(),
            'version': self.get_version(),
            'description': self.get_description(),
            'dependencies': self.get_dependencies(),
            'interfaces': [iface.__name__ for iface in self.get_interfaces()],
            'status': self.status.value,
            'config': self.get_config()
        }
    
    def __str__(self) -> str:
        """String representation of the plugin."""
        return f"{self.get_name()} v{self.get_version()} ({self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed representation of the plugin."""
        return f"<{self.__class__.__name__}: {self.get_name()} v{self.get_version()}>"
