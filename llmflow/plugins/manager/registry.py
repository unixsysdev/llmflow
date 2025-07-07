"""
Plugin Registry Module

This module provides a registry system for managing plugin instances and their interfaces.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from collections import defaultdict
from datetime import datetime

from ..interfaces.base import Plugin
from ..interfaces.transport import ITransportProtocol
from ..interfaces.security import ISecurityProvider
from ..interfaces.serialization import IMessageSerializer
from ..interfaces.storage import IStorageProvider
from ..interfaces.monitoring import IMonitoringProvider

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PluginRegistry:
    """
    Registry for managing plugin instances and their interfaces.
    
    This class provides a centralized registry for all loaded plugins,
    allowing lookup by name, interface type, or other criteria.
    """
    
    def __init__(self):
        # Main plugin registry
        self.plugins: Dict[str, Plugin] = {}
        
        # Interface-based registries
        self.transport_providers: Dict[str, ITransportProtocol] = {}
        self.security_providers: Dict[str, ISecurityProvider] = {}
        self.serializers: Dict[str, IMessageSerializer] = {}
        self.storage_providers: Dict[str, IStorageProvider] = {}
        self.monitoring_providers: Dict[str, IMonitoringProvider] = {}
        
        # Interface mappings
        self.interface_plugins: Dict[Type, Dict[str, Plugin]] = {
            ITransportProtocol: {},
            ISecurityProvider: {},
            IMessageSerializer: {},
            IStorageProvider: {},
            IMonitoringProvider: {}
        }
        
        # Metadata
        self.registration_times: Dict[str, datetime] = {}
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_plugin(self, plugin: Plugin) -> None:
        """
        Register a plugin in the registry.
        
        Args:
            plugin: Plugin instance to register
        """
        plugin_name = plugin.get_name()
        
        if plugin_name in self.plugins:
            logger.warning(f"Plugin {plugin_name} is already registered, updating...")
        
        # Register in main registry
        self.plugins[plugin_name] = plugin
        self.registration_times[plugin_name] = datetime.utcnow()
        self.plugin_metadata[plugin_name] = plugin.get_metadata()
        
        # Register in interface-specific registries
        self._register_by_interfaces(plugin)
        
        logger.info(f"Registered plugin: {plugin_name}")
    
    def _register_by_interfaces(self, plugin: Plugin) -> None:
        """Register plugin by its implemented interfaces."""
        plugin_name = plugin.get_name()
        interfaces = plugin.get_interfaces()
        
        for interface_type in interfaces:
            if interface_type == ITransportProtocol:
                self.transport_providers[plugin_name] = plugin
                self.interface_plugins[ITransportProtocol][plugin_name] = plugin
            elif interface_type == ISecurityProvider:
                self.security_providers[plugin_name] = plugin
                self.interface_plugins[ISecurityProvider][plugin_name] = plugin
            elif interface_type == IMessageSerializer:
                self.serializers[plugin_name] = plugin
                self.interface_plugins[IMessageSerializer][plugin_name] = plugin
            elif interface_type == IStorageProvider:
                self.storage_providers[plugin_name] = plugin
                self.interface_plugins[IStorageProvider][plugin_name] = plugin
            elif interface_type == IMonitoringProvider:
                self.monitoring_providers[plugin_name] = plugin
                self.interface_plugins[IMonitoringProvider][plugin_name] = plugin
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin from the registry.
        
        Args:
            plugin_name: Name of the plugin to unregister
            
        Returns:
            True if unregistered successfully, False if not found
        """
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin {plugin_name} is not registered")
            return False
        
        # Remove from main registry
        plugin = self.plugins.pop(plugin_name)
        self.registration_times.pop(plugin_name, None)
        self.plugin_metadata.pop(plugin_name, None)
        
        # Remove from interface-specific registries
        self._unregister_by_interfaces(plugin_name, plugin)
        
        logger.info(f"Unregistered plugin: {plugin_name}")
        return True
    
    def _unregister_by_interfaces(self, plugin_name: str, plugin: Plugin) -> None:
        """Unregister plugin from interface-specific registries."""
        interfaces = plugin.get_interfaces()
        
        for interface_type in interfaces:
            if interface_type == ITransportProtocol:
                self.transport_providers.pop(plugin_name, None)
                self.interface_plugins[ITransportProtocol].pop(plugin_name, None)
            elif interface_type == ISecurityProvider:
                self.security_providers.pop(plugin_name, None)
                self.interface_plugins[ISecurityProvider].pop(plugin_name, None)
            elif interface_type == IMessageSerializer:
                self.serializers.pop(plugin_name, None)
                self.interface_plugins[IMessageSerializer].pop(plugin_name, None)
            elif interface_type == IStorageProvider:
                self.storage_providers.pop(plugin_name, None)
                self.interface_plugins[IStorageProvider].pop(plugin_name, None)
            elif interface_type == IMonitoringProvider:
                self.monitoring_providers.pop(plugin_name, None)
                self.interface_plugins[IMonitoringProvider].pop(plugin_name, None)
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        Get a plugin by name.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_interface(self, interface_type: Type[T]) -> List[T]:
        """
        Get all plugins that implement a specific interface.
        
        Args:
            interface_type: Interface type to search for
            
        Returns:
            List of plugins implementing the interface
        """
        if interface_type not in self.interface_plugins:
            return []
        
        return list(self.interface_plugins[interface_type].values())
    
    def get_transport_provider(self, name: Optional[str] = None) -> Optional[ITransportProtocol]:
        """
        Get a transport provider by name, or the first available one.
        
        Args:
            name: Optional name of the transport provider
            
        Returns:
            Transport provider instance or None
        """
        if name:
            return self.transport_providers.get(name)
        elif self.transport_providers:
            return next(iter(self.transport_providers.values()))
        return None
    
    def get_security_provider(self, name: Optional[str] = None) -> Optional[ISecurityProvider]:
        """
        Get a security provider by name, or the first available one.
        
        Args:
            name: Optional name of the security provider
            
        Returns:
            Security provider instance or None
        """
        if name:
            return self.security_providers.get(name)
        elif self.security_providers:
            return next(iter(self.security_providers.values()))
        return None
    
    def get_serializer(self, name: Optional[str] = None) -> Optional[IMessageSerializer]:
        """
        Get a serializer by name, or the first available one.
        
        Args:
            name: Optional name of the serializer
            
        Returns:
            Serializer instance or None
        """
        if name:
            return self.serializers.get(name)
        elif self.serializers:
            return next(iter(self.serializers.values()))
        return None
    
    def get_storage_provider(self, name: Optional[str] = None) -> Optional[IStorageProvider]:
        """
        Get a storage provider by name, or the first available one.
        
        Args:
            name: Optional name of the storage provider
            
        Returns:
            Storage provider instance or None
        """
        if name:
            return self.storage_providers.get(name)
        elif self.storage_providers:
            return next(iter(self.storage_providers.values()))
        return None
    
    def get_monitoring_provider(self, name: Optional[str] = None) -> Optional[IMonitoringProvider]:
        """
        Get a monitoring provider by name, or the first available one.
        
        Args:
            name: Optional name of the monitoring provider
            
        Returns:
            Monitoring provider instance or None
        """
        if name:
            return self.monitoring_providers.get(name)
        elif self.monitoring_providers:
            return next(iter(self.monitoring_providers.values()))
        return None
    
    def list_plugins(self) -> List[str]:
        """
        Get a list of all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self.plugins.keys())
    
    def list_plugins_by_interface(self, interface_type: Type) -> List[str]:
        """
        Get a list of plugin names that implement a specific interface.
        
        Args:
            interface_type: Interface type to search for
            
        Returns:
            List of plugin names
        """
        if interface_type not in self.interface_plugins:
            return []
        
        return list(self.interface_plugins[interface_type].keys())
    
    def get_plugin_count(self) -> int:
        """
        Get the total number of registered plugins.
        
        Returns:
            Number of registered plugins
        """
        return len(self.plugins)
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin metadata or None if not found
        """
        return self.plugin_metadata.get(plugin_name)
    
    def get_all_plugin_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all registered plugins.
        
        Returns:
            Dictionary mapping plugin names to their metadata
        """
        return self.plugin_metadata.copy()
    
    def get_registration_time(self, plugin_name: str) -> Optional[datetime]:
        """
        Get the registration time for a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Registration time or None if not found
        """
        return self.registration_times.get(plugin_name)
    
    def is_plugin_registered(self, plugin_name: str) -> bool:
        """
        Check if a plugin is registered.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            True if registered, False otherwise
        """
        return plugin_name in self.plugins
    
    def find_plugins_by_criteria(self, **criteria) -> List[Plugin]:
        """
        Find plugins that match specific criteria.
        
        Args:
            **criteria: Search criteria (e.g., version="1.0", author="John")
            
        Returns:
            List of matching plugins
        """
        matching_plugins = []
        
        for plugin_name, plugin in self.plugins.items():
            metadata = self.plugin_metadata.get(plugin_name, {})
            
            # Check if plugin matches all criteria
            matches = True
            for key, value in criteria.items():
                if key not in metadata or metadata[key] != value:
                    matches = False
                    break
            
            if matches:
                matching_plugins.append(plugin)
        
        return matching_plugins
    
    def get_interface_statistics(self) -> Dict[str, int]:
        """
        Get statistics about registered interfaces.
        
        Returns:
            Dictionary mapping interface names to counts
        """
        stats = {}
        
        for interface_type, plugins in self.interface_plugins.items():
            interface_name = interface_type.__name__
            stats[interface_name] = len(plugins)
        
        return stats
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.
        
        Returns:
            Dictionary containing registry statistics
        """
        return {
            'total_plugins': len(self.plugins),
            'registered_plugins': list(self.plugins.keys()),
            'interface_statistics': self.get_interface_statistics(),
            'transport_providers': len(self.transport_providers),
            'security_providers': len(self.security_providers),
            'serializers': len(self.serializers),
            'storage_providers': len(self.storage_providers),
            'monitoring_providers': len(self.monitoring_providers),
            'oldest_registration': min(self.registration_times.values()) if self.registration_times else None,
            'newest_registration': max(self.registration_times.values()) if self.registration_times else None
        }
    
    def clear_registry(self) -> None:
        """Clear all registered plugins."""
        logger.warning("Clearing entire plugin registry")
        
        self.plugins.clear()
        self.transport_providers.clear()
        self.security_providers.clear()
        self.serializers.clear()
        self.storage_providers.clear()
        self.monitoring_providers.clear()
        
        for interface_plugins in self.interface_plugins.values():
            interface_plugins.clear()
        
        self.registration_times.clear()
        self.plugin_metadata.clear()


# Global registry instance
_global_registry: Optional[PluginRegistry] = None


def get_global_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.
    
    Returns:
        Global plugin registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def register_plugin(plugin: Plugin) -> None:
    """
    Register a plugin in the global registry.
    
    Args:
        plugin: Plugin to register
    """
    get_global_registry().register_plugin(plugin)


def unregister_plugin(plugin_name: str) -> bool:
    """
    Unregister a plugin from the global registry.
    
    Args:
        plugin_name: Name of the plugin to unregister
        
    Returns:
        True if unregistered successfully, False if not found
    """
    return get_global_registry().unregister_plugin(plugin_name)


def get_plugin(plugin_name: str) -> Optional[Plugin]:
    """
    Get a plugin from the global registry.
    
    Args:
        plugin_name: Name of the plugin
        
    Returns:
        Plugin instance or None if not found
    """
    return get_global_registry().get_plugin(plugin_name)
