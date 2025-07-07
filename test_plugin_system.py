"""
Test Plugin System

Quick test to verify the plugin system is working correctly.
"""

import asyncio
import sys
sys.path.insert(0, '.')

async def test_plugin_system():
    """Test the plugin system functionality."""
    print("🔍 Testing Plugin System...")
    
    try:
        # Test plugin interfaces
        from llmflow.plugins.interfaces.base import Plugin
        from llmflow.plugins.interfaces.transport import ITransportProtocol
        from llmflow.plugins.manager.plugin_manager import PluginManager
        from llmflow.plugins.manager.registry import get_global_registry
        
        print("  ✓ Plugin interfaces imported successfully")
        
        # Test example plugin
        from llmflow.plugins.examples.dummy_transport import DummyUDPTransport
        
        plugin = DummyUDPTransport({"test": "config"})
        print(f"  ✓ Created dummy plugin: {plugin.get_name()} v{plugin.get_version()}")
        
        # Test plugin initialization
        await plugin.initialize({"port": 8888})
        print(f"  ✓ Plugin initialized")
        
        # Test plugin manager
        manager = PluginManager()
        print(f"  ✓ Plugin manager created")
        
        # Test plugin registry
        registry = get_global_registry()
        print(f"  ✓ Plugin registry accessed")
        
        # Test health check
        health = await plugin.health_check()
        print(f"  ✓ Plugin health check: {health}")
        
        # Test plugin status
        status = plugin.get_status()
        print(f"  ✓ Plugin status: {status}")
        
        await plugin.shutdown()
        print(f"  ✓ Plugin shutdown completed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Plugin system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🚀 Plugin System Test\n")
    
    success = await test_plugin_system()
    
    print(f"\nResult: {'✅ Plugin system works!' if success else '❌ Plugin system failed'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
