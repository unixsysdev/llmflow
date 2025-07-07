"""
Plugin Manager Module

This module provides the core plugin management functionality for LLMFlow,
including plugin discovery, loading, validation, and lifecycle management.
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..interfaces.base import Plugin, PluginStatus, PluginError, PluginInitializationError
from .registry import PluginRegistry
from .validator import PluginValidator

logger = logging.getLogger(__name__)


@dataclass
class PluginDiscoveryConfig:
    """Configuration for plugin discovery."""
    search_paths: List[str] = field(default_factory=lambda: [
        "./plugins",
        "~/.llmflow/plugins", 
        "/usr/local/lib/llmflow/plugins"
    ])
    auto_load: bool = True
    recursive_search: bool = True
    file_patterns: List[str] = field(default_factory=lambda: ["*.py", "plugin.py", "__init__.py"])
    excluded_dirs: List[str] = field(default_factory=lambda: ["__pycache__", ".git", "tests"])


@dataclass
class PluginLoadConfig:
    """Configuration for plugin loading."""
    validation_strict: bool = True
    dependency_resolution: bool = True
    hot_reload: bool = False
    rollback_on_failure: bool = True
    timeout_seconds: int = 30


class PluginManager:
    """
    Central plugin manager for LLMFlow.
    
    This class handles plugin discovery, loading, validation, and lifecycle management.
    """
    
    def __init__(self, 
                 discovery_config: Optional[PluginDiscoveryConfig] = None,
                 load_config: Optional[PluginLoadConfig] = None):
        self.discovery_config = discovery_config or PluginDiscoveryConfig()
        self.load_config = load_config or PluginLoadConfig()
        self.registry = PluginRegistry()
        self.validator = PluginValidator()
        
        # Plugin lifecycle tracking
        self.loaded_plugins: Dict[str, Plugin] = {}
        self.plugin_modules: Dict[str, Any] = {}
        self.load_order: List[str] = []
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Event callbacks
        self.on_plugin_loaded: List[Callable[[Plugin], None]] = []
        self.on_plugin_unloaded: List[Callable[[str], None]] = []
        self.on_plugin_error: List[Callable[[str, Exception], None]] = []
        
        self._lock = asyncio.Lock()
    
    async def discover_plugins(self) -> List[str]:
        """
        Discover plugins in the configured search paths.
        
        Returns:
            List of discovered plugin module paths
        """
        logger.info("Starting plugin discovery...")
        discovered_plugins = []
        
        for search_path in self.discovery_config.search_paths:
            expanded_path = Path(search_path).expanduser().resolve()
            
            if not expanded_path.exists():
                logger.warning(f"Plugin search path does not exist: {expanded_path}")
                continue
            
            logger.debug(f"Searching for plugins in: {expanded_path}")
            
            if self.discovery_config.recursive_search:
                plugin_files = self._find_plugin_files_recursive(expanded_path)
            else:
                plugin_files = self._find_plugin_files_direct(expanded_path)
            
            discovered_plugins.extend(plugin_files)
        
        logger.info(f"Discovered {len(discovered_plugins)} plugin files")
        return discovered_plugins
    
    def _find_plugin_files_recursive(self, path: Path) -> List[str]:
        """Find plugin files recursively."""
        plugin_files = []
        
        for root, dirs, files in os.walk(path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.discovery_config.excluded_dirs]
            
            root_path = Path(root)
            
            for file in files:
                if any(file.endswith(pattern.replace("*", "")) for pattern in self.discovery_config.file_patterns):
                    if file.endswith(".py"):
                        module_path = self._file_to_module_path(root_path / file, path)
                        if module_path:
                            plugin_files.append(module_path)
        
        return plugin_files
    
    def _find_plugin_files_direct(self, path: Path) -> List[str]:
        """Find plugin files directly in path."""
        plugin_files = []
        
        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix == ".py":
                module_path = self._file_to_module_path(file_path, path)
                if module_path:
                    plugin_files.append(module_path)
        
        return plugin_files
    
    def _file_to_module_path(self, file_path: Path, base_path: Path) -> Optional[str]:
        """Convert file path to module path."""
        try:
            relative_path = file_path.relative_to(base_path)
            module_path = str(relative_path.with_suffix("")).replace(os.sep, ".")
            return module_path
        except ValueError:
            return None
    
    async def load_plugin(self, module_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load a plugin from module path.
        
        Args:
            module_path: Module path to load
            config: Optional plugin configuration
            
        Returns:
            True if loading was successful, False otherwise
        """
        async with self._lock:
            try:
                logger.info(f"Loading plugin from: {module_path}")
                
                # Import the module
                module = importlib.import_module(module_path)
                
                # Find plugin classes in the module
                plugin_classes = self._find_plugin_classes(module)
                
                if not plugin_classes:
                    logger.warning(f"No plugin classes found in module: {module_path}")
                    return False
                
                # Load each plugin class
                for plugin_class in plugin_classes:
                    await self._load_plugin_class(plugin_class, module_path, config)
                
                self.plugin_modules[module_path] = module
                logger.info(f"Successfully loaded plugin module: {module_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load plugin from {module_path}: {e}")
                self._notify_plugin_error(module_path, e)
                return False
    
    def _find_plugin_classes(self, module: Any) -> List[Type[Plugin]]:
        """Find plugin classes in a module."""
        plugin_classes = []
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, Plugin) and 
                obj is not Plugin):
                plugin_classes.append(obj)
        
        return plugin_classes
    
    async def _load_plugin_class(self, 
                                plugin_class: Type[Plugin], 
                                module_path: str, 
                                config: Optional[Dict[str, Any]] = None) -> None:
        """Load a specific plugin class."""
        try:
            # Create plugin instance
            plugin_instance = plugin_class(config or {})
            plugin_name = plugin_instance.get_name()
            
            # Validate plugin
            if self.load_config.validation_strict:
                validation_result = await self.validator.validate_plugin(plugin_instance)
                if not validation_result.is_valid:
                    raise PluginInitializationError(f"Plugin validation failed: {validation_result.errors}")
            
            # Check dependencies
            if self.load_config.dependency_resolution:
                await self._resolve_dependencies(plugin_instance)
            
            # Initialize plugin
            await plugin_instance.initialize(config or {})
            
            # Register plugin
            self.registry.register_plugin(plugin_instance)
            self.loaded_plugins[plugin_name] = plugin_instance
            self.load_order.append(plugin_name)
            
            # Update dependency graph
            self._update_dependency_graph(plugin_instance)
            
            logger.info(f"Successfully loaded plugin: {plugin_name}")
            self._notify_plugin_loaded(plugin_instance)
            
        except Exception as e:
            logger.error(f"Failed to load plugin class {plugin_class.__name__}: {e}")
            raise
    
    async def _resolve_dependencies(self, plugin: Plugin) -> None:
        """Resolve plugin dependencies."""
        dependencies = plugin.get_dependencies()
        
        for dep in dependencies:
            if dep not in self.loaded_plugins:
                logger.warning(f"Plugin {plugin.get_name()} depends on {dep}, but it's not loaded")
                
                # Try to load the dependency
                dep_loaded = await self._try_load_dependency(dep)
                if not dep_loaded:
                    raise PluginInitializationError(f"Failed to load dependency: {dep}")
    
    async def _try_load_dependency(self, dependency_name: str) -> bool:
        """Try to load a dependency plugin."""
        # This is a simplified implementation
        # In a real implementation, you'd have a dependency resolver
        # that can find and load dependencies from various sources
        logger.info(f"Attempting to load dependency: {dependency_name}")
        return False
    
    def _update_dependency_graph(self, plugin: Plugin) -> None:
        """Update the dependency graph."""
        plugin_name = plugin.get_name()
        dependencies = set(plugin.get_dependencies())
        self.dependency_graph[plugin_name] = dependencies
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if unloading was successful, False otherwise
        """
        async with self._lock:
            try:
                if plugin_name not in self.loaded_plugins:
                    logger.warning(f"Plugin {plugin_name} is not loaded")
                    return False
                
                plugin = self.loaded_plugins[plugin_name]
                
                # Check if other plugins depend on this one
                dependents = self._get_dependents(plugin_name)
                if dependents:
                    logger.warning(f"Cannot unload {plugin_name}: other plugins depend on it: {dependents}")
                    return False
                
                # Stop and shutdown the plugin
                await plugin.stop()
                await plugin.shutdown()
                
                # Remove from tracking
                del self.loaded_plugins[plugin_name]
                self.load_order.remove(plugin_name)
                del self.dependency_graph[plugin_name]
                
                # Unregister from registry
                self.registry.unregister_plugin(plugin_name)
                
                logger.info(f"Successfully unloaded plugin: {plugin_name}")
                self._notify_plugin_unloaded(plugin_name)
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload plugin {plugin_name}: {e}")
                self._notify_plugin_error(plugin_name, e)
                return False
    
    def _get_dependents(self, plugin_name: str) -> List[str]:
        """Get plugins that depend on the given plugin."""
        dependents = []
        
        for name, deps in self.dependency_graph.items():
            if plugin_name in deps:
                dependents.append(name)
        
        return dependents
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin.
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            True if reloading was successful, False otherwise
        """
        if not self.load_config.hot_reload:
            logger.warning("Hot reload is disabled")
            return False
        
        # Find the module path for this plugin
        module_path = self._find_module_path_for_plugin(plugin_name)
        if not module_path:
            logger.error(f"Cannot find module path for plugin: {plugin_name}")
            return False
        
        # Get current config
        current_config = None
        if plugin_name in self.loaded_plugins:
            current_config = self.loaded_plugins[plugin_name].get_config()
        
        # Unload the plugin
        unload_success = await self.unload_plugin(plugin_name)
        if not unload_success:
            logger.error(f"Failed to unload plugin for reload: {plugin_name}")
            return False
        
        # Reload the module
        try:
            if module_path in sys.modules:
                importlib.reload(sys.modules[module_path])
        except Exception as e:
            logger.error(f"Failed to reload module {module_path}: {e}")
            return False
        
        # Load the plugin again
        load_success = await self.load_plugin(module_path, current_config)
        if not load_success:
            logger.error(f"Failed to load plugin after reload: {plugin_name}")
            return False
        
        logger.info(f"Successfully reloaded plugin: {plugin_name}")
        return True
    
    def _find_module_path_for_plugin(self, plugin_name: str) -> Optional[str]:
        """Find the module path for a plugin."""
        for module_path, module in self.plugin_modules.items():
            plugin_classes = self._find_plugin_classes(module)
            for plugin_class in plugin_classes:
                try:
                    temp_instance = plugin_class()
                    if temp_instance.get_name() == plugin_name:
                        return module_path
                except Exception:
                    continue
        return None
    
    async def start_all_plugins(self) -> None:
        """Start all loaded plugins."""
        logger.info("Starting all plugins...")
        
        for plugin_name in self.load_order:
            plugin = self.loaded_plugins[plugin_name]
            try:
                await plugin.start()
                logger.info(f"Started plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to start plugin {plugin_name}: {e}")
                self._notify_plugin_error(plugin_name, e)
    
    async def stop_all_plugins(self) -> None:
        """Stop all loaded plugins."""
        logger.info("Stopping all plugins...")
        
        # Stop in reverse order
        for plugin_name in reversed(self.load_order):
            plugin = self.loaded_plugins[plugin_name]
            try:
                await plugin.stop()
                logger.info(f"Stopped plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to stop plugin {plugin_name}: {e}")
                self._notify_plugin_error(plugin_name, e)
    
    async def shutdown_all_plugins(self) -> None:
        """Shutdown all loaded plugins."""
        logger.info("Shutting down all plugins...")
        
        # Shutdown in reverse order
        for plugin_name in reversed(self.load_order):
            plugin = self.loaded_plugins[plugin_name]
            try:
                await plugin.shutdown()
                logger.info(f"Shutdown plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to shutdown plugin {plugin_name}: {e}")
                self._notify_plugin_error(plugin_name, e)
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name."""
        return self.loaded_plugins.get(plugin_name)
    
    def get_all_plugins(self) -> Dict[str, Plugin]:
        """Get all loaded plugins."""
        return self.loaded_plugins.copy()
    
    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """Check if a plugin is loaded."""
        return plugin_name in self.loaded_plugins
    
    def get_plugin_status(self, plugin_name: str) -> Optional[PluginStatus]:
        """Get the status of a plugin."""
        plugin = self.loaded_plugins.get(plugin_name)
        return plugin.get_status() if plugin else None
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all plugins."""
        results = {}
        
        for plugin_name, plugin in self.loaded_plugins.items():
            try:
                is_healthy = await plugin.health_check()
                results[plugin_name] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for plugin {plugin_name}: {e}")
                results[plugin_name] = False
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        return {
            'total_plugins': len(self.loaded_plugins),
            'loaded_plugins': list(self.loaded_plugins.keys()),
            'load_order': self.load_order.copy(),
            'dependency_graph': {k: list(v) for k, v in self.dependency_graph.items()},
            'discovery_config': {
                'search_paths': self.discovery_config.search_paths,
                'auto_load': self.discovery_config.auto_load,
                'recursive_search': self.discovery_config.recursive_search
            },
            'load_config': {
                'validation_strict': self.load_config.validation_strict,
                'dependency_resolution': self.load_config.dependency_resolution,
                'hot_reload': self.load_config.hot_reload
            }
        }
    
    # Event notification methods
    def _notify_plugin_loaded(self, plugin: Plugin) -> None:
        """Notify that a plugin was loaded."""
        for callback in self.on_plugin_loaded:
            try:
                callback(plugin)
            except Exception as e:
                logger.error(f"Error in plugin loaded callback: {e}")
    
    def _notify_plugin_unloaded(self, plugin_name: str) -> None:
        """Notify that a plugin was unloaded."""
        for callback in self.on_plugin_unloaded:
            try:
                callback(plugin_name)
            except Exception as e:
                logger.error(f"Error in plugin unloaded callback: {e}")
    
    def _notify_plugin_error(self, plugin_name: str, error: Exception) -> None:
        """Notify that a plugin error occurred."""
        for callback in self.on_plugin_error:
            try:
                callback(plugin_name, error)
            except Exception as e:
                logger.error(f"Error in plugin error callback: {e}")
