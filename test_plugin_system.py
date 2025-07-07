"""
Plugin System Tests

This module contains comprehensive tests for the LLMFlow plugin system.
"""

import asyncio
import pytest
from pathlib import Path
from typing import Dict, Any

from llmflow.plugins.manager import PluginManager, PluginDiscoveryConfig, PluginLoadConfig
from llmflow.plugins.manager.registry import PluginRegistry
from llmflow.plugins.manager.validator import PluginValidator
from llmflow.plugins.config.loader import ConfigurationLoader
from llmflow.plugins.examples import DummyUDPTransport, DummySecurityProvider, DummyJSONSerializer


class TestPluginSystem:
    """Test the plugin system functionality."""
    
    def test_plugin_registry(self):
        """Test plugin registry functionality."""
        print("Testing plugin registry...")
        
        registry = PluginRegistry()
        
        # Test plugin registration
        plugin = DummyUDPTransport()
        registry.register_plugin(plugin)
        
        assert registry.is_plugin_registered(plugin.get_name())
        assert registry.get_plugin_count() == 1
        
        # Test plugin retrieval
        retrieved = registry.get_plugin(plugin.get_name())
        assert retrieved is plugin
        
        # Test plugin unregistration
        success = registry.unregister_plugin(plugin.get_name())
        assert success
        assert registry.get_plugin_count() == 0
        
        print("✓ Plugin registry tests passed")
    
    async def test_plugin_validation(self):
        """Test plugin validation functionality."""
        print("Testing plugin validation...")
        
        validator = PluginValidator()
        
        # Test valid plugin
        plugin = DummyUDPTransport()
        await plugin.initialize({})
        
        result = await validator.validate_plugin(plugin)
        assert result.is_valid, f"Validation failed: {result.errors}"
        
        # Test security plugin
        security_plugin = DummySecurityProvider()
        await security_plugin.initialize({})
        
        result = await validator.validate_plugin(security_plugin)
        assert result.is_valid, f"Security plugin validation failed: {result.errors}"
        
        # Test serializer plugin
        serializer_plugin = DummyJSONSerializer()
        await serializer_plugin.initialize({})
        
        result = await validator.validate_plugin(serializer_plugin)
        assert result.is_valid, f"Serializer plugin validation failed: {result.errors}"
        
        print("✓ Plugin validation tests passed")
    
    def test_configuration_loader(self):
        """Test configuration loader functionality."""
        print("Testing configuration loader...")
        
        loader = ConfigurationLoader()
        
        # Test default configuration
        default_config = {
            'core': {
                'transport': 'udp',
                'security': 'jwt'
            },
            'plugins': {
                'udp': {
                    'port': 8080
                }
            }
        }
        
        loader.add_default_config(default_config)
        
        # Test environment variables
        import os
        os.environ['LLMFLOW_CORE_DEBUG'] = 'true'
        os.environ['LLMFLOW_PLUGINS_UDP_TIMEOUT'] = '30'
        
        loader.add_environment_variables()
        
        # Test configuration merging
        config = loader.merge_configuration()
        assert config['core']['transport'] == 'udp'
        assert config['core']['debug'] == True
        assert config['plugins']['udp']['timeout'] == 30
        
        # Test plugin-specific configuration
        plugin_config = loader.get_plugin_config('udp')
        assert plugin_config['transport'] == 'udp'
        assert plugin_config['timeout'] == 30
        
        print("✓ Configuration loader tests passed")
    
    async def test_plugin_lifecycle(self):
        """Test plugin lifecycle management."""
        print("Testing plugin lifecycle...")
        
        # Test transport plugin lifecycle
        transport = DummyUDPTransport()
        assert transport.get_status().value == 'uninitialized'
        
        await transport.initialize({})
        assert transport.get_status().value == 'initialized'
        
        await transport.start()
        assert transport.get_status().value == 'running'
        
        health = await transport.health_check()
        assert health == True
        
        await transport.stop()
        assert transport.get_status().value == 'stopped'
        
        await transport.shutdown()
        
        # Test security plugin lifecycle
        security = DummySecurityProvider()
        await security.initialize({})
        await security.start()
        
        health = await security.health_check()
        assert health == True
        
        await security.stop()
        await security.shutdown()
        
        print("✓ Plugin lifecycle tests passed")
    
    async def test_plugin_functionality(self):
        """Test actual plugin functionality."""
        print("Testing plugin functionality...")
        
        # Test transport plugin
        transport = DummyUDPTransport()
        await transport.initialize({'buffer_size': 1024})
        await transport.start()
        
        # Test options
        await transport.set_option('timeout', 10)
        timeout = await transport.get_option('timeout')
        assert timeout == 10
        
        # Test stats
        stats = await transport.get_stats()
        assert 'transport_type' in stats
        assert stats['transport_type'] == 'udp'
        
        await transport.shutdown()
        
        # Test security plugin
        security = DummySecurityProvider()
        await security.initialize({})
        await security.start()
        
        # Test authentication
        credentials = {'username': 'admin', 'password': 'admin123'}
        token = await security.authenticate(credentials)
        assert token is not None
        assert token.is_valid()
        
        # Test token validation
        validated_token = await security.validate_token(token.token)
        assert validated_token is not None
        assert validated_token.token == token.token
        
        # Test authorization
        context = await security.create_security_context('admin', token)
        authorized = await security.authorize(context, 'test_resource', 'read')
        assert authorized == True
        
        await security.shutdown()
        
        # Test serializer plugin
        serializer = DummyJSONSerializer()
        await serializer.initialize({})
        await serializer.start()
        
        # Test serialization
        test_data = {'message': 'Hello, World!', 'number': 42}
        serialized = await serializer.serialize(test_data)
        assert isinstance(serialized, bytes)
        
        # Test deserialization
        deserialized = await serializer.deserialize(serialized)
        assert deserialized == test_data
        
        # Test metadata
        metadata = await serializer.get_metadata(serialized)
        assert metadata.format.value == 'json'
        
        await serializer.shutdown()
        
        print("✓ Plugin functionality tests passed")
    
    async def test_plugin_manager(self):
        """Test plugin manager functionality."""
        print("Testing plugin manager...")
        
        # Create plugin manager
        discovery_config = PluginDiscoveryConfig(
            search_paths=['./llmflow/plugins/examples'],
            auto_load=False
        )
        
        load_config = PluginLoadConfig(
            validation_strict=True,
            dependency_resolution=False
        )
        
        manager = PluginManager(discovery_config, load_config)
        
        # Test plugin discovery
        plugins = await manager.discover_plugins()
        assert len(plugins) > 0
        
        # Test manual plugin loading
        transport = DummyUDPTransport()
        await transport.initialize({})
        
        manager.registry.register_plugin(transport)
        manager.loaded_plugins[transport.get_name()] = transport
        
        # Test plugin retrieval
        retrieved = manager.get_plugin(transport.get_name())
        assert retrieved is transport
        
        # Test plugin status
        status = manager.get_plugin_status(transport.get_name())
        assert status is not None
        
        # Test health check
        health_results = await manager.health_check_all()
        assert transport.get_name() in health_results
        
        # Test stats
        stats = manager.get_stats()
        assert 'total_plugins' in stats
        assert stats['total_plugins'] == 1
        
        print("✓ Plugin manager tests passed")


async def test_plugin_integration():
    """Test plugin integration."""
    print("Testing plugin integration...")
    
    # Create plugins
    transport = DummyUDPTransport()
    security = DummySecurityProvider()
    serializer = DummyJSONSerializer()
    
    # Initialize all plugins
    await transport.initialize({'buffer_size': 1024})
    await security.initialize({})
    await serializer.initialize({})
    
    # Start all plugins
    await transport.start()
    await security.start()
    await serializer.start()
    
    # Test integration: serialize data, encrypt it, send it
    test_data = {'user': 'admin', 'action': 'test'}
    
    # Serialize data
    serialized_data = await serializer.serialize(test_data)
    
    # Encrypt data
    encrypted_data = await security.encrypt(serialized_data)
    
    # Sign data
    signature = await security.sign(encrypted_data)
    
    # Verify signature
    verified = await security.verify(encrypted_data, signature)
    assert verified == True
    
    # Decrypt data
    decrypted_data = await security.decrypt(encrypted_data)
    
    # Deserialize data
    final_data = await serializer.deserialize(decrypted_data)
    assert final_data == test_data
    
    # Test transport (simulated)
    transport_stats = await transport.get_stats()
    assert 'transport_type' in transport_stats
    
    # Shutdown all plugins
    await transport.shutdown()
    await security.shutdown()
    await serializer.shutdown()
    
    print("✓ Plugin integration tests passed")


async def main():
    """Run all plugin system tests."""
    print("🚀 Starting plugin system tests...\n")
    
    try:
        test_system = TestPluginSystem()
        
        # Run synchronous tests
        test_system.test_plugin_registry()
        test_system.test_configuration_loader()
        
        # Run asynchronous tests
        await test_system.test_plugin_validation()
        await test_system.test_plugin_lifecycle()
        await test_system.test_plugin_functionality()
        await test_system.test_plugin_manager()
        
        # Run integration tests
        await test_plugin_integration()
        
        print("\n🎉 All plugin system tests passed!")
        print("✅ Plugin system is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Plugin system tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
