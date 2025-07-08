"""
Configuration Loader Module

This module provides configuration loading and management functionality for LLMFlow plugins.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationSource:
    """Represents a configuration source."""
    source_type: str  # 'file', 'env', 'default'
    source_path: Optional[str] = None
    priority: int = 0
    data: Dict[str, Any] = field(default_factory=dict)


class ConfigurationLoader:
    """
    Configuration loader for LLMFlow plugins.
    
    This class handles loading configuration from multiple sources including
    YAML files, environment variables, and default values.
    """
    
    def __init__(self):
        self.sources: List[ConfigurationSource] = []
        self.merged_config: Dict[str, Any] = {}
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        
        # Environment variable prefix
        self.env_prefix = "LLMFLOW_"
    
    def add_config_file(self, file_path: Union[str, Path], priority: int = 0) -> bool:
        """
        Add a configuration file as a source.
        
        Args:
            file_path: Path to the configuration file
            priority: Priority of this source (higher = more important)
            
        Returns:
            True if file was loaded successfully, False otherwise
        """
        try:
            path = Path(file_path).expanduser().resolve()
            
            if not path.exists():
                logger.warning(f"Configuration file does not exist: {path}")
                return False
            
            # Load configuration based on file extension
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = self._load_yaml_file(path)
            elif path.suffix.lower() == '.json':
                data = self._load_json_file(path)
            else:
                logger.warning(f"Unsupported configuration file format: {path.suffix}")
                return False
            
            source = ConfigurationSource(
                source_type='file',
                source_path=str(path),
                priority=priority,
                data=data
            )
            
            self.sources.append(source)
            logger.info(f"Added configuration file: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration file {file_path}: {e}")
            return False
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading YAML file {file_path}: {e}")
            raise
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if data is not None else {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            raise
    
    def add_environment_variables(self, priority: int = 10) -> None:
        """
        Add environment variables as a configuration source.
        
        Args:
            priority: Priority of this source (higher = more important)
        """
        try:
            env_data = {}
            
            for key, value in os.environ.items():
                if key.startswith(self.env_prefix):
                    # Remove prefix and convert to lowercase
                    config_key = key[len(self.env_prefix):].lower()
                    
                    # Convert nested keys (e.g., LLMFLOW_PLUGIN_NAME -> plugin.name)
                    nested_keys = config_key.split('_')
                    
                    # Build nested dictionary
                    current = env_data
                    for nested_key in nested_keys[:-1]:
                        if nested_key not in current:
                            current[nested_key] = {}
                        current = current[nested_key]
                    
                    # Set the final value
                    current[nested_keys[-1]] = self._convert_env_value(value)
            
            if env_data:
                source = ConfigurationSource(
                    source_type='env',
                    priority=priority,
                    data=env_data
                )
                
                self.sources.append(source)
                logger.info(f"Added environment variables configuration with {len(env_data)} keys")
        
        except Exception as e:
            logger.error(f"Failed to load environment variables: {e}")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value to appropriate type."""
        # Try to convert to boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try to convert to integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try to convert to JSON (for lists, objects)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Return as string
        return value
    
    def add_default_config(self, default_config: Dict[str, Any], priority: int = -10) -> None:
        """
        Add default configuration.
        
        Args:
            default_config: Default configuration dictionary
            priority: Priority of this source (usually lowest)
        """
        try:
            source = ConfigurationSource(
                source_type='default',
                priority=priority,
                data=default_config.copy()
            )
            
            self.sources.append(source)
            logger.info("Added default configuration")
        
        except Exception as e:
            logger.error(f"Failed to add default configuration: {e}")
    
    def merge_configuration(self) -> Dict[str, Any]:
        """
        Merge all configuration sources by priority.
        
        Returns:
            Merged configuration dictionary
        """
        try:
            # Sort sources by priority (highest first)
            sorted_sources = sorted(self.sources, key=lambda x: x.priority, reverse=True)
            
            merged = {}
            
            # Merge configurations (higher priority overwrites lower)
            for source in reversed(sorted_sources):  # Start with lowest priority
                merged = self._deep_merge_dict(merged, source.data)
            
            self.merged_config = merged
            logger.info(f"Merged configuration from {len(self.sources)} sources")
            return merged
        
        except Exception as e:
            logger.error(f"Failed to merge configuration: {e}")
            return {}
    
    def _deep_merge_dict(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config(self, key: Optional[str] = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            
        Returns:
            Configuration value or None if not found
        """
        if not self.merged_config:
            self.merge_configuration()
        
        if key is None:
            return self.merged_config
        
        # Handle nested keys
        keys = key.split('.')
        current = self.merged_config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin configuration dictionary
        """
        if plugin_name in self.config_cache:
            return self.config_cache[plugin_name]
        
        # Get plugin-specific configuration
        plugin_config = self.get_config(f'plugins.{plugin_name}') or {}
        
        # Get core configuration that applies to all plugins
        core_config = self.get_config('core') or {}
        
        # Merge core and plugin-specific configuration
        merged_plugin_config = self._deep_merge_dict(core_config, plugin_config)
        
        # Cache the result
        self.config_cache[plugin_name] = merged_plugin_config
        
        return merged_plugin_config
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            value: Configuration value
        """
        if not self.merged_config:
            self.merge_configuration()
        
        # Handle nested keys
        keys = key.split('.')
        current = self.merged_config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        
        # Clear cache since configuration changed
        self.config_cache.clear()
    
    def validate_configuration(self, schema: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against a schema.
        
        Args:
            schema: Configuration schema
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            # Simple validation - check required fields
            required_fields = schema.get('required', [])
            
            for field in required_fields:
                if self.get_config(field) is None:
                    errors.append(f"Required configuration field missing: {field}")
            
            # Check field types
            properties = schema.get('properties', {})
            
            for field, field_schema in properties.items():
                value = self.get_config(field)
                if value is not None:
                    expected_type = field_schema.get('type')
                    if expected_type and not self._check_type(value, expected_type):
                        errors.append(f"Configuration field {field} has wrong type: expected {expected_type}")
        
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
        
        return errors
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True
    
    def reload_configuration(self) -> bool:
        """
        Reload configuration from all sources.
        
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            # Clear current configuration
            self.merged_config.clear()
            self.config_cache.clear()
            
            # Reload file-based sources
            for source in self.sources:
                if source.source_type == 'file' and source.source_path:
                    path = Path(source.source_path)
                    if path.exists():
                        if path.suffix.lower() in ['.yaml', '.yml']:
                            source.data = self._load_yaml_file(path)
                        elif path.suffix.lower() == '.json':
                            source.data = self._load_json_file(path)
            
            # Reload environment variables
            env_sources = [s for s in self.sources if s.source_type == 'env']
            if env_sources:
                # Remove old env sources
                self.sources = [s for s in self.sources if s.source_type != 'env']
                # Add updated env variables
                self.add_environment_variables(env_sources[0].priority)
            
            # Merge configuration
            self.merge_configuration()
            
            logger.info("Configuration reloaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Configuration summary dictionary
        """
        return {
            'sources': [
                {
                    'type': source.source_type,
                    'path': source.source_path,
                    'priority': source.priority,
                    'keys': len(source.data) if isinstance(source.data, dict) else 0
                }
                for source in self.sources
            ],
            'total_sources': len(self.sources),
            'merged_keys': len(self.merged_config) if isinstance(self.merged_config, dict) else 0,
            'cached_plugins': len(self.config_cache),
            'env_prefix': self.env_prefix
        }


# Global configuration loader instance
_global_loader: Optional[ConfigurationLoader] = None


def get_global_config_loader() -> ConfigurationLoader:
    """
    Get the global configuration loader instance.
    
    Returns:
        Global configuration loader
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigurationLoader()
    return _global_loader


def load_default_configuration() -> Dict[str, Any]:
    """
    Load default LLMFlow configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'core': {
            'serializer': 'messagepack',
            'transport': 'udp',
            'security': 'jwt',
            'storage': 'in_memory',
            'monitoring': 'prometheus'
        },
        'plugins': {
            'discovery': {
                'paths': [
                    './plugins',
                    '~/.llmflow/plugins',
                    '/usr/local/lib/llmflow/plugins'
                ],
                'auto_load': True,
                'recursive_search': True
            },
            'loading': {
                'validation_strict': True,
                'dependency_resolution': True,
                'hot_reload': False,
                'timeout_seconds': 30
            }
        },
        'queue': {
            'default_size': 1000,
            'max_message_size': 65536,
            'timeout_seconds': 5
        },
        'conductor': {
            'health_check_interval': 30,
            'metrics_collection_interval': 10
        },
        'master': {
            'optimization_interval': 300,
            'consensus_timeout': 10
        },
        'llm': {
            'provider': 'openai',
            'api_key': None,  # Set via OPENAI_API_KEY environment variable
            'model': 'gpt-4',
            'fallback_model': 'gpt-3.5-turbo',
            'max_tokens': 4000,
            'temperature': 0.1,
            'timeout_seconds': 30,
            'retry_attempts': 3,
            'optimization': {
                'enabled': True,
                'confidence_threshold': 0.7,
                'auto_apply_threshold': 0.9,
                'validation_required': True
            }
        }
    }

