"""
Plugin Validator Module

This module provides validation functionality for LLMFlow plugins,
ensuring they meet interface requirements and compatibility standards.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Type, Set
from dataclasses import dataclass
from datetime import datetime

from ..interfaces.base import Plugin, PluginValidationError
from ..interfaces.transport import ITransportProtocol
from ..interfaces.security import ISecurityProvider
from ..interfaces.serialization import IMessageSerializer
from ..interfaces.storage import IStorageProvider
from ..interfaces.monitoring import IMonitoringProvider

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of plugin validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validation_time: datetime
    
    def __post_init__(self):
        if self.validation_time is None:
            self.validation_time = datetime.utcnow()
    
    @classmethod
    def success(cls, warnings: Optional[List[str]] = None) -> 'ValidationResult':
        """Create a successful validation result."""
        return cls(
            is_valid=True,
            errors=[],
            warnings=warnings or [],
            validation_time=datetime.utcnow()
        )
    
    @classmethod
    def failure(cls, errors: List[str], warnings: Optional[List[str]] = None) -> 'ValidationResult':
        """Create a failed validation result."""
        return cls(
            is_valid=False,
            errors=errors,
            warnings=warnings or [],
            validation_time=datetime.utcnow()
        )
    
    def add_error(self, error: str) -> None:
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'validation_time': self.validation_time.isoformat()
        }


class PluginValidator:
    """
    Validator for LLMFlow plugins.
    
    This class provides comprehensive validation of plugin implementations,
    ensuring they meet interface requirements and compatibility standards.
    """
    
    def __init__(self):
        self.interface_validators = {
            ITransportProtocol: self._validate_transport_interface,
            ISecurityProvider: self._validate_security_interface,
            IMessageSerializer: self._validate_serializer_interface,
            IStorageProvider: self._validate_storage_interface,
            IMonitoringProvider: self._validate_monitoring_interface
        }
    
    async def validate_plugin(self, plugin: Plugin) -> ValidationResult:
        """
        Validate a plugin comprehensively.
        
        Args:
            plugin: Plugin to validate
            
        Returns:
            Validation result
        """
        logger.debug(f"Validating plugin: {plugin.get_name()}")
        
        result = ValidationResult.success()
        
        # Basic plugin validation
        self._validate_base_plugin(plugin, result)
        
        # Interface-specific validation
        self._validate_plugin_interfaces(plugin, result)
        
        # Metadata validation
        self._validate_plugin_metadata(plugin, result)
        
        # Configuration validation
        self._validate_plugin_configuration(plugin, result)
        
        # Dependency validation
        self._validate_plugin_dependencies(plugin, result)
        
        # Security validation
        self._validate_plugin_security(plugin, result)
        
        logger.debug(f"Validation completed for {plugin.get_name()}: {'✓' if result.is_valid else '✗'}")
        return result
    
    def _validate_base_plugin(self, plugin: Plugin, result: ValidationResult) -> None:
        """Validate basic plugin requirements."""
        try:
            # Check required methods
            name = plugin.get_name()
            if not name or not isinstance(name, str):
                result.add_error("Plugin name must be a non-empty string")
            
            version = plugin.get_version()
            if not version or not isinstance(version, str):
                result.add_error("Plugin version must be a non-empty string")
            
            description = plugin.get_description()
            if not description or not isinstance(description, str):
                result.add_error("Plugin description must be a non-empty string")
            
            dependencies = plugin.get_dependencies()
            if not isinstance(dependencies, list):
                result.add_error("Plugin dependencies must be a list")
            
            interfaces = plugin.get_interfaces()
            if not isinstance(interfaces, list):
                result.add_error("Plugin interfaces must be a list")
            
            # Check if plugin implements required abstract methods
            abstract_methods = self._get_abstract_methods(Plugin)
            for method_name in abstract_methods:
                if not hasattr(plugin, method_name):
                    result.add_error(f"Plugin missing required method: {method_name}")
                elif not callable(getattr(plugin, method_name)):
                    result.add_error(f"Plugin method {method_name} is not callable")
            
        except Exception as e:
            result.add_error(f"Error validating base plugin: {str(e)}")
    
    def _validate_plugin_interfaces(self, plugin: Plugin, result: ValidationResult) -> None:
        """Validate plugin interface implementations."""
        try:
            interfaces = plugin.get_interfaces()
            
            for interface_type in interfaces:
                if not isinstance(interface_type, type):
                    result.add_error(f"Invalid interface type: {interface_type}")
                    continue
                
                # Check if plugin actually implements the interface
                if not isinstance(plugin, interface_type):
                    result.add_error(f"Plugin does not implement declared interface: {interface_type.__name__}")
                    continue
                
                # Run interface-specific validation
                validator = self.interface_validators.get(interface_type)
                if validator:
                    validator(plugin, result)
                else:
                    result.add_warning(f"No specific validator for interface: {interface_type.__name__}")
        
        except Exception as e:
            result.add_error(f"Error validating plugin interfaces: {str(e)}")
    
    def _validate_plugin_metadata(self, plugin: Plugin, result: ValidationResult) -> None:
        """Validate plugin metadata."""
        try:
            metadata = plugin.get_metadata()
            
            if not isinstance(metadata, dict):
                result.add_error("Plugin metadata must be a dictionary")
                return
            
            # Check required metadata fields
            required_fields = ['name', 'version', 'description', 'dependencies', 'interfaces']
            for field in required_fields:
                if field not in metadata:
                    result.add_error(f"Missing required metadata field: {field}")
            
            # Validate metadata values
            if 'name' in metadata and not isinstance(metadata['name'], str):
                result.add_error("Metadata 'name' must be a string")
            
            if 'version' in metadata and not isinstance(metadata['version'], str):
                result.add_error("Metadata 'version' must be a string")
            
            if 'dependencies' in metadata and not isinstance(metadata['dependencies'], list):
                result.add_error("Metadata 'dependencies' must be a list")
            
            if 'interfaces' in metadata and not isinstance(metadata['interfaces'], list):
                result.add_error("Metadata 'interfaces' must be a list")
        
        except Exception as e:
            result.add_error(f"Error validating plugin metadata: {str(e)}")
    
    def _validate_plugin_configuration(self, plugin: Plugin, result: ValidationResult) -> None:
        """Validate plugin configuration."""
        try:
            config = plugin.get_config()
            
            if not isinstance(config, dict):
                result.add_error("Plugin configuration must be a dictionary")
                return
            
            # Try to validate configuration using plugin's own validation
            try:
                plugin.validate_config(config)
            except Exception as e:
                result.add_error(f"Plugin configuration validation failed: {str(e)}")
        
        except Exception as e:
            result.add_error(f"Error validating plugin configuration: {str(e)}")
    
    def _validate_plugin_dependencies(self, plugin: Plugin, result: ValidationResult) -> None:
        """Validate plugin dependencies."""
        try:
            dependencies = plugin.get_dependencies()
            
            for dep in dependencies:
                if not isinstance(dep, str):
                    result.add_error(f"Dependency must be a string: {dep}")
                elif not dep.strip():
                    result.add_error("Dependency cannot be empty")
        
        except Exception as e:
            result.add_error(f"Error validating plugin dependencies: {str(e)}")
    
    def _validate_plugin_security(self, plugin: Plugin, result: ValidationResult) -> None:
        """Validate plugin security requirements."""
        try:
            # Check if plugin has any security-sensitive methods
            security_methods = ['authenticate', 'authorize', 'encrypt', 'decrypt', 'sign', 'verify']
            
            for method_name in security_methods:
                if hasattr(plugin, method_name):
                    result.add_warning(f"Plugin implements security-sensitive method: {method_name}")
            
            # Check if plugin accesses file system
            file_methods = ['open', 'read', 'write', 'delete']
            for method_name in file_methods:
                if hasattr(plugin, method_name):
                    result.add_warning(f"Plugin may access file system: {method_name}")
        
        except Exception as e:
            result.add_error(f"Error validating plugin security: {str(e)}")
    
    def _validate_transport_interface(self, plugin: ITransportProtocol, result: ValidationResult) -> None:
        """Validate transport protocol interface implementation."""
        try:
            required_methods = [
                'get_transport_type', 'bind', 'connect', 'send', 'receive', 
                'close', 'is_connected', 'get_connection_info', 'set_option', 
                'get_option', 'get_stats'
            ]
            
            self._check_required_methods(plugin, required_methods, result, "ITransportProtocol")
        
        except Exception as e:
            result.add_error(f"Error validating transport interface: {str(e)}")
    
    def _validate_security_interface(self, plugin: ISecurityProvider, result: ValidationResult) -> None:
        """Validate security provider interface implementation."""
        try:
            required_methods = [
                'get_authentication_method', 'authenticate', 'validate_token', 
                'refresh_token', 'revoke_token', 'authorize', 'encrypt', 'decrypt',
                'sign', 'verify', 'generate_key', 'rotate_key', 'create_security_context',
                'get_user_permissions', 'audit_log'
            ]
            
            self._check_required_methods(plugin, required_methods, result, "ISecurityProvider")
        
        except Exception as e:
            result.add_error(f"Error validating security interface: {str(e)}")
    
    def _validate_serializer_interface(self, plugin: IMessageSerializer, result: ValidationResult) -> None:
        """Validate serializer interface implementation."""
        try:
            required_methods = [
                'get_format', 'get_content_type', 'get_file_extension',
                'supports_schema_evolution', 'supports_compression',
                'get_supported_compression_types', 'serialize', 'deserialize',
                'get_metadata', 'validate_schema', 'register_schema',
                'get_schema', 'get_schema_version', 'set_schema_version',
                'compress', 'decompress', 'get_stats'
            ]
            
            self._check_required_methods(plugin, required_methods, result, "IMessageSerializer")
        
        except Exception as e:
            result.add_error(f"Error validating serializer interface: {str(e)}")
    
    def _validate_storage_interface(self, plugin: IStorageProvider, result: ValidationResult) -> None:
        """Validate storage provider interface implementation."""
        try:
            required_methods = [
                'get_storage_type', 'get_consistency_level', 'connect', 'disconnect',
                'is_connected', 'put', 'get', 'delete', 'exists', 'list_keys',
                'query', 'count', 'batch_put', 'batch_get', 'batch_delete',
                'clear', 'backup', 'restore', 'get_stats', 'health_check',
                'transaction_begin', 'transaction_commit', 'transaction_rollback',
                'create_index', 'drop_index', 'list_indexes'
            ]
            
            self._check_required_methods(plugin, required_methods, result, "IStorageProvider")
        
        except Exception as e:
            result.add_error(f"Error validating storage interface: {str(e)}")
    
    def _validate_monitoring_interface(self, plugin: IMonitoringProvider, result: ValidationResult) -> None:
        """Validate monitoring provider interface implementation."""
        try:
            required_methods = [
                'get_provider_name', 'get_supported_metric_types', 'connect',
                'disconnect', 'is_connected', 'record_metric', 'record_counter',
                'record_gauge', 'record_histogram', 'record_timer', 'create_timer',
                'send_alert', 'resolve_alert', 'get_active_alerts', 'query_metrics',
                'get_metric_names', 'get_metric_tags', 'create_dashboard',
                'update_dashboard', 'delete_dashboard', 'get_dashboards',
                'create_alert_rule', 'update_alert_rule', 'delete_alert_rule',
                'get_alert_rules', 'get_stats', 'health_check', 'batch_record_metrics',
                'set_global_tags', 'get_global_tags'
            ]
            
            self._check_required_methods(plugin, required_methods, result, "IMonitoringProvider")
        
        except Exception as e:
            result.add_error(f"Error validating monitoring interface: {str(e)}")
    
    def _check_required_methods(self, 
                               plugin: Any, 
                               required_methods: List[str], 
                               result: ValidationResult, 
                               interface_name: str) -> None:
        """Check if plugin implements required methods."""
        for method_name in required_methods:
            if not hasattr(plugin, method_name):
                result.add_error(f"{interface_name} missing required method: {method_name}")
            elif not callable(getattr(plugin, method_name)):
                result.add_error(f"{interface_name} method {method_name} is not callable")
    
    def _get_abstract_methods(self, cls: Type) -> Set[str]:
        """Get abstract methods from a class."""
        abstract_methods = set()
        
        for name, method in inspect.getmembers(cls):
            if inspect.isfunction(method) and getattr(method, '__isabstractmethod__', False):
                abstract_methods.add(name)
        
        return abstract_methods
    
    def validate_plugin_compatibility(self, plugin1: Plugin, plugin2: Plugin) -> ValidationResult:
        """
        Validate compatibility between two plugins.
        
        Args:
            plugin1: First plugin
            plugin2: Second plugin
            
        Returns:
            Validation result
        """
        result = ValidationResult.success()
        
        try:
            # Check for interface conflicts
            interfaces1 = set(plugin1.get_interfaces())
            interfaces2 = set(plugin2.get_interfaces())
            
            # Check for interface overlaps (potential conflicts)
            overlaps = interfaces1.intersection(interfaces2)
            for interface in overlaps:
                result.add_warning(f"Both plugins implement interface: {interface.__name__}")
            
            # Check for dependency conflicts
            deps1 = set(plugin1.get_dependencies())
            deps2 = set(plugin2.get_dependencies())
            
            # Check if plugins depend on each other (circular dependency)
            if plugin1.get_name() in deps2 and plugin2.get_name() in deps1:
                result.add_error("Circular dependency detected between plugins")
            
            # Check for conflicting configurations
            config1 = plugin1.get_config()
            config2 = plugin2.get_config()
            
            # Look for conflicting port usage
            if 'port' in config1 and 'port' in config2:
                if config1['port'] == config2['port']:
                    result.add_error("Both plugins configured to use same port")
        
        except Exception as e:
            result.add_error(f"Error validating plugin compatibility: {str(e)}")
        
        return result
    
    def validate_plugin_version_compatibility(self, plugin: Plugin, required_version: str) -> ValidationResult:
        """
        Validate plugin version compatibility.
        
        Args:
            plugin: Plugin to validate
            required_version: Required version string
            
        Returns:
            Validation result
        """
        result = ValidationResult.success()
        
        try:
            plugin_version = plugin.get_version()
            
            # Simple version comparison (could be enhanced with semantic versioning)
            if plugin_version != required_version:
                result.add_warning(f"Plugin version {plugin_version} does not match required version {required_version}")
        
        except Exception as e:
            result.add_error(f"Error validating version compatibility: {str(e)}")
        
        return result
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary containing validation statistics
        """
        return {
            'supported_interfaces': list(self.interface_validators.keys()),
            'validator_count': len(self.interface_validators),
            'last_validation_time': datetime.utcnow().isoformat()
        }
