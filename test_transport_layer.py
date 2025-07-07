"""
Transport Layer Tests

This module contains comprehensive tests for the LLMFlow transport layer implementations.
"""

import asyncio
import pytest
from typing import Dict, Any

from llmflow.transport import (
    BaseTransport, TransportConfig, TransportState,
    UDPTransport, UDPTransportPlugin, UDPConfig,
    TCPTransport, TCPTransportPlugin, TCPConfig,
    WebSocketTransport, WebSocketTransportPlugin, WebSocketConfig
)
from llmflow.plugins.interfaces.transport import TransportType


class TestTransportLayer:
    """Test the transport layer functionality."""
    
    def test_transport_config(self):
        """Test transport configuration."""
        print("Testing transport configuration...")
        
        # Test base config
        config = TransportConfig(
            address="127.0.0.1",
            port=8080,
            timeout=30.0,
            buffer_size=4096
        )
        
        assert config.address == "127.0.0.1"
        assert config.port == 8080
        assert config.timeout == 30.0
        assert config.buffer_size == 4096
        
        # Test UDP config
        udp_config = UDPConfig(
            address="127.0.0.1",
            port=8081,
            max_retries=5,
            ack_timeout=2.0
        )
        
        assert udp_config.address == "127.0.0.1"
        assert udp_config.port == 8081
        assert udp_config.max_retries == 5
        assert udp_config.ack_timeout == 2.0
        
        # Test TCP config
        tcp_config = TCPConfig(
            address="127.0.0.1",
            port=8082,
            connection_pool_size=20,
            tcp_nodelay=True
        )
        
        assert tcp_config.address == "127.0.0.1"
        assert tcp_config.port == 8082
        assert tcp_config.connection_pool_size == 20
        assert tcp_config.tcp_nodelay == True
        
        # Test WebSocket config
        ws_config = WebSocketConfig(
            address="127.0.0.1",
            port=8083,
            ping_interval=60.0,
            auto_ping=True
        )
        
        assert ws_config.address == "127.0.0.1"
        assert ws_config.port == 8083
        assert ws_config.ping_interval == 60.0
        assert ws_config.auto_ping == True
        
        print("✓ Transport configuration tests passed")
    
    async def test_udp_transport(self):
        """Test UDP transport implementation."""
        print("Testing UDP transport...")
        
        # Create UDP transport
        config = UDPConfig(
            address="127.0.0.1",
            port=18080,
            max_retries=3
        )
        
        transport = UDPTransport(config)
        
        # Test transport type
        assert transport.get_transport_type() == TransportType.UDP
        
        # Test binding
        success = await transport.bind("127.0.0.1", 18080)
        assert success == True
        assert transport.is_connected() == True
        
        # Test options
        await transport.set_option("buffer_size", 8192)
        buffer_size = await transport.get_option("buffer_size")
        assert buffer_size == 8192
        
        # Test stats
        stats = transport.get_stats()
        assert "transport_type" in stats
        assert stats["transport_type"] == "udp"
        
        # Close transport
        await transport.close()
        assert transport.is_connected() == False
        
        print("✓ UDP transport tests passed")
    
    async def test_tcp_transport(self):
        """Test TCP transport implementation."""
        print("Testing TCP transport...")
        
        # Create TCP transport
        config = TCPConfig(
            address="127.0.0.1",
            port=18081,
            connection_pool_size=5,
            tcp_nodelay=True
        )
        
        transport = TCPTransport(config)
        
        # Test transport type
        assert transport.get_transport_type() == TransportType.TCP
        
        # Test binding (server mode)
        success = await transport.bind("127.0.0.1", 18081)
        assert success == True
        assert transport.is_connected() == True
        
        # Test stats
        stats = await transport.get_stats()
        assert "transport_type" in stats
        assert stats["transport_type"] == "tcp"
        assert "connection_pool" in stats
        
        # Close transport
        await transport.close()
        assert transport.is_connected() == False
        
        print("✓ TCP transport tests passed")
    
    async def test_websocket_transport(self):
        """Test WebSocket transport implementation."""
        print("Testing WebSocket transport...")
        
        # Create WebSocket transport
        config = WebSocketConfig(
            address="127.0.0.1",
            port=18082,
            ping_interval=30.0,
            auto_ping=True
        )
        
        transport = WebSocketTransport(config)
        
        # Test transport type
        assert transport.get_transport_type() == TransportType.WEBSOCKET
        
        # Test binding (server mode)
        success = await transport.bind("127.0.0.1", 18082)
        assert success == True
        assert transport.is_connected() == True
        
        # Test stats
        stats = await transport.get_stats()
        assert "transport_type" in stats
        assert stats["transport_type"] == "websocket"
        assert "client_connections" in stats
        
        # Close transport
        await transport.close()
        assert transport.is_connected() == False
        
        print("✓ WebSocket transport tests passed")
    
    async def test_transport_plugins(self):
        """Test transport plugin implementations."""
        print("Testing transport plugins...")
        
        # Test UDP plugin
        udp_plugin = UDPTransportPlugin({"port": 18083})
        assert udp_plugin.get_name() == "udp_transport"
        assert udp_plugin.get_version() == "1.0.0"
        assert len(udp_plugin.get_dependencies()) == 0
        
        await udp_plugin.initialize({"port": 18083})
        await udp_plugin.start()
        
        # Test transport type
        assert udp_plugin.get_transport_type() == TransportType.UDP
        
        # Test health check
        health = await udp_plugin.health_check()
        assert health == True
        
        await udp_plugin.shutdown()
        
        # Test TCP plugin
        tcp_plugin = TCPTransportPlugin({"port": 18084})
        assert tcp_plugin.get_name() == "tcp_transport"
        assert tcp_plugin.get_version() == "1.0.0"
        
        await tcp_plugin.initialize({"port": 18084})
        await tcp_plugin.start()
        
        # Test transport type
        assert tcp_plugin.get_transport_type() == TransportType.TCP
        
        await tcp_plugin.shutdown()
        
        # Test WebSocket plugin
        ws_plugin = WebSocketTransportPlugin({"port": 18085})
        assert ws_plugin.get_name() == "websocket_transport"
        assert ws_plugin.get_version() == "1.0.0"
        
        await ws_plugin.initialize({"port": 18085})
        await ws_plugin.start()
        
        # Test transport type
        assert ws_plugin.get_transport_type() == TransportType.WEBSOCKET
        
        await ws_plugin.shutdown()
        
        print("✓ Transport plugin tests passed")
    
    async def test_transport_events(self):
        """Test transport event handling."""
        print("Testing transport events...")
        
        # Create UDP transport
        config = UDPConfig(address="127.0.0.1", port=18086)
        transport = UDPTransport(config)
        
        # Track events
        events_received = []
        
        def event_handler(event_name, event_data):
            events_received.append((event_name, event_data))
        
        # Add event handler
        transport.add_event_handler("bound", event_handler)
        transport.add_event_handler("closed", event_handler)
        
        # Test binding (should trigger event)
        await transport.bind("127.0.0.1", 18086)
        
        # Test closing (should trigger event)
        await transport.close()
        
        # Verify events were received
        assert len(events_received) >= 2
        event_names = [event[0] for event in events_received]
        assert "bound" in event_names
        assert "closed" in event_names
        
        print("✓ Transport event tests passed")
    
    async def test_transport_error_handling(self):
        """Test transport error handling."""
        print("Testing transport error handling...")
        
        # Test invalid port binding
        config = UDPConfig(address="127.0.0.1", port=99999)  # Invalid port
        transport = UDPTransport(config)
        
        # This should fail gracefully
        success = await transport.bind("127.0.0.1", 99999)
        assert success == False
        
        # Test sending without connection
        success = await transport.send(b"test data")
        assert success == False
        
        # Test receiving without connection
        result = await transport.receive(timeout=0.1)
        assert result is None
        
        print("✓ Transport error handling tests passed")
    
    async def test_transport_integration(self):
        """Test transport integration with other components."""
        print("Testing transport integration...")
        
        # Create different transport types
        udp_config = UDPConfig(address="127.0.0.1", port=18087)
        tcp_config = TCPConfig(address="127.0.0.1", port=18088)
        ws_config = WebSocketConfig(address="127.0.0.1", port=18089)
        
        udp_transport = UDPTransport(udp_config)
        tcp_transport = TCPTransport(tcp_config)
        ws_transport = WebSocketTransport(ws_config)
        
        # Test all transports can bind to different ports
        udp_success = await udp_transport.bind("127.0.0.1", 18087)
        tcp_success = await tcp_transport.bind("127.0.0.1", 18088)
        ws_success = await ws_transport.bind("127.0.0.1", 18089)
        
        assert udp_success == True
        assert tcp_success == True
        assert ws_success == True
        
        # Test all transports report as connected
        assert udp_transport.is_connected() == True
        assert tcp_transport.is_connected() == True
        assert ws_transport.is_connected() == True
        
        # Get stats from all transports
        udp_stats = udp_transport.get_stats()
        tcp_stats = await tcp_transport.get_stats()
        ws_stats = await ws_transport.get_stats()
        
        assert udp_stats["transport_type"] == "udp"
        assert tcp_stats["transport_type"] == "tcp"
        assert ws_stats["transport_type"] == "websocket"
        
        # Close all transports
        await udp_transport.close()
        await tcp_transport.close()
        await ws_transport.close()
        
        # Verify all are disconnected
        assert udp_transport.is_connected() == False
        assert tcp_transport.is_connected() == False
        assert ws_transport.is_connected() == False
        
        print("✓ Transport integration tests passed")


async def test_transport_performance():
    """Test transport performance characteristics."""
    print("Testing transport performance...")
    
    # Create UDP transport for performance testing
    config = UDPConfig(address="127.0.0.1", port=18090)
    transport = UDPTransport(config)
    
    await transport.bind("127.0.0.1", 18090)
    
    # Test multiple operations
    import time
    start_time = time.time()
    
    operations = 100
    for i in range(operations):
        await transport.set_option("buffer_size", 1024 + i)
        await transport.get_option("buffer_size")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should be reasonably fast
    assert duration < 1.0  # Less than 1 second for 100 operations
    
    # Get final stats
    stats = transport.get_stats()
    assert "stats" in stats
    
    await transport.close()
    
    print(f"✓ Transport performance tests passed ({operations} operations in {duration:.3f}s)")


async def main():
    """Run all transport layer tests."""
    print("🚀 Starting transport layer tests...\n")
    
    try:
        test_system = TestTransportLayer()
        
        # Run synchronous tests
        test_system.test_transport_config()
        
        # Run asynchronous tests
        await test_system.test_udp_transport()
        await test_system.test_tcp_transport()
        await test_system.test_websocket_transport()
        await test_system.test_transport_plugins()
        await test_system.test_transport_events()
        await test_system.test_transport_error_handling()
        await test_system.test_transport_integration()
        
        # Run performance tests
        await test_transport_performance()
        
        print("\n🎉 All transport layer tests passed!")
        print("✅ Transport layer is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Transport layer tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
