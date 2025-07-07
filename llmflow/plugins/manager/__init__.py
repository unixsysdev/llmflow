"""
Plugin Manager Module

This module provides comprehensive plugin management functionality for LLMFlow,
including plugin discovery, loading, validation, and lifecycle management.
"""

from .plugin_manager import PluginManager, PluginDiscoveryConfig, PluginLoadConfig
from .registry import PluginRegistry, get_global_registry, register_plugin, unregister_plugin, get_plugin
from .validator import PluginValidator, ValidationResult

__all__ = [
    # Plugin Manager
    'PluginManager',
    'PluginDiscoveryConfig', 
    'PluginLoadConfig',
    
    # Registry
    'PluginRegistry',
    'get_global_registry',
    'register_plugin',
    'unregister_plugin',
    'get_plugin',
    
    # Validator
    'PluginValidator',
    'ValidationResult',
]
