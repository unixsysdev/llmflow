#!/usr/bin/env python3
"""
Debug Transport Layer Issues

This script helps identify specific issues with the UDP/TCP transport implementation
without requiring the full test framework.
"""

import asyncio
import sys
import os

# Add project to path
sys.path.insert(0, '.')

def debug_imports():
    """Debug import issues step by step."""
    print("🔍 Debugging imports...")
    
    # Test basic transport imports
    try:
        print("  Testing transport base imports...")
        from llmflow.transport.base import BaseTransport, TransportConfig, TransportState
        print("  ✓ Base transport imports successful")
    except Exception as e:
        print(f"  ❌ Base transport import failed: {e}")
        return False
    
    try:
        print("  Testing UDP transport imports...")
        from llmflow.transport.udp.transport import UDPTransport, UDPConfig
        print("  ✓ UDP transport imports successful")
    except Exception as e:
        print(f"  ❌ UDP transport import failed: {e}")
        return False
    
    try:
        print("  Testing TCP transport imports...")
        from llmflow.transport.tcp.transport import TCPTransport, TCPConfig
        print("  ✓ TCP transport imports successful")
    except Exception as e:
        print(f"  ❌ TCP transport import failed: {e}")
        return False
    
    return True

async def test_udp_basic():
    """Test basic UDP functionality."""
    print("\n🔍 Testing UDP basic functionality...")
    
    try:
        from llmflow.transport.udp.transport import UDPTransport, UDPConfig
        from llmflow.plugins.interfaces.transport import TransportType
        
        # Create UDP config
        config = UDPConfig(
            address="127.0.0.1",
            port=18100,
            max_retries=3,
            ack_timeout=1.0
        )
        
        print(f"  Created UDP config: {config.address}:{config.port}")
        
        # Create transport
        transport = UDPTransport(config)
        print(f"  ✓ Created UDP transport")
        
        # Test transport type
        transport_type = transport.get_transport_type()
        print(f"  ✓ Transport type: {transport_type}")
        assert transport_type == TransportType.UDP
        
        # Test binding
        print("  Testing UDP bind...")
        success = await transport.bind("127.0.0.1", 18100)
        print(f"  Bind result: {success}")
        
        if success:
            print("  ✓ UDP bind successful")
            
            # Test connection status
            is_connected = transport.is_connected()
            print(f"  Connection status: {is_connected}")
            
            # Close transport
            await transport.close()
            print("  ✓ UDP transport closed")
            
            return True
        else:
            print("  ❌ UDP bind failed")
            return False
            
    except Exception as e:
        print(f"  ❌ UDP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_tcp_basic():
    """Test basic TCP functionality."""
    print("\n🔍 Testing TCP basic functionality...")
    
    try:
        from llmflow.transport.tcp.transport import TCPTransport, TCPConfig
        from llmflow.plugins.interfaces.transport import TransportType
        
        # Create TCP config
        config = TCPConfig(
            address="127.0.0.1",
            port=18101,
            connection_pool_size=5,
            tcp_nodelay=True
        )
        
        print(f"  Created TCP config: {config.address}:{config.port}")
        
        # Create transport
        transport = TCPTransport(config)
        print(f"  ✓ Created TCP transport")
        
        # Test transport type
        transport_type = transport.get_transport_type()
        print(f"  ✓ Transport type: {transport_type}")
        assert transport_type == TransportType.TCP
        
        # Test binding
        print("  Testing TCP bind...")
        success = await transport.bind("127.0.0.1", 18101)
        print(f"  Bind result: {success}")
        
        if success:
            print("  ✓ TCP bind successful")
            
            # Test connection status
            is_connected = transport.is_connected()
            print(f"  Connection status: {is_connected}")
            
            # Get stats
            stats = await transport.get_stats()
            print(f"  Stats: {stats.get('transport_type', 'unknown')}")
            
            # Close transport
            await transport.close()
            print("  ✓ TCP transport closed")
            
            return True
        else:
            print("  ❌ TCP bind failed")
            return False
            
    except Exception as e:
        print(f"  ❌ TCP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_message_sending():
    """Test actual message sending between transports."""
    print("\n🔍 Testing message sending...")
    
    try:
        from llmflow.transport.udp.transport import UDPTransport, UDPConfig, ReliabilityMode
        
        # Test both with and without reliability
        test_cases = [
            ("Without Reliability", ReliabilityMode.NONE),
            ("With Acknowledgment", ReliabilityMode.ACKNOWLEDGMENT)
        ]
        
        all_passed = True
        
        for test_name, reliability_mode in test_cases:
            print(f"\n  Testing {test_name}...")
            
            # Create server transport (bind mode)
            server_config = UDPConfig(
                address="127.0.0.1", 
                port=18102, 
                reliability_mode=reliability_mode,
                ack_timeout=0.5,  # Shorter timeout for testing
                max_retries=2
            )
            server_transport = UDPTransport(server_config)
            
            # Create client transport (connect mode)
            client_config = UDPConfig(
                address="127.0.0.1", 
                port=18102,
                reliability_mode=reliability_mode,
                ack_timeout=0.5,
                max_retries=2
            )
            client_transport = UDPTransport(client_config)
            
            try:
                # Bind server
                server_success = await server_transport.bind("127.0.0.1", 18102)
                if not server_success:
                    print(f"    ❌ Failed to bind server")
                    all_passed = False
                    continue
                
                print(f"    ✓ Server transport bound successfully")
                
                # Connect client
                client_success = await client_transport.connect("127.0.0.1", 18102)
                if not client_success:
                    print(f"    ❌ Failed to connect client")
                    await server_transport.close()
                    all_passed = False
                    continue
                
                print(f"    ✓ Client transport connected successfully")
                
                # Test sending message
                test_message = b"Hello, Reliable World!"
                print(f"    Attempting to send: {test_message}")
                
                send_success = await client_transport.send(test_message)
                print(f"    Send result: {send_success}")
                
                if send_success:
                    print(f"    ✓ Message sent successfully")
                    
                    # Try to receive (with timeout)
                    try:
                        result = await asyncio.wait_for(
                            server_transport.receive(timeout=2.0),
                            timeout=3.0
                        )
                        
                        if result:
                            received_data, sender = result
                            print(f"    ✓ Received: {received_data}")
                            print(f"    ✓ Sender: {sender}")
                            success = received_data == test_message
                            print(f"    Message match: {success}")
                            if not success:
                                all_passed = False
                        else:
                            print(f"    ❌ No message received")
                            all_passed = False
                    except asyncio.TimeoutError:
                        print(f"    ❌ Receive timeout")
                        all_passed = False
                else:
                    print(f"    ❌ Failed to send message")
                    all_passed = False
                
            finally:
                # Clean up
                await server_transport.close()
                await client_transport.close()
                # Wait a bit to ensure ports are released
                await asyncio.sleep(0.1)
        
        return all_passed
        
    except Exception as e:
        print(f"  ❌ Message sending test failed: {e}")
        import traceback
        traceback.print_exc()
        return False




def check_missing_dependencies():
    """Check for missing dependencies that might cause issues."""
    print("\n🔍 Checking dependencies...")
    
    missing_deps = []
    
    try:
        import msgpack
        print("  ✓ msgpack available")
    except ImportError:
        print("  ❌ msgpack missing")
        missing_deps.append("msgpack")
    
    try:
        import pydantic
        print("  ✓ pydantic available")
    except ImportError:
        print("  ❌ pydantic missing")
        missing_deps.append("pydantic")
    
    try:
        import asyncio
        print("  ✓ asyncio available")
    except ImportError:
        print("  ❌ asyncio missing")
        missing_deps.append("asyncio")
    
    if missing_deps:
        print(f"\n  Missing dependencies: {', '.join(missing_deps)}")
        print("  Install with: pip install " + " ".join(missing_deps))
        return False
    else:
        print("  ✓ All dependencies available")
        return True

async def main():
    """Main debug function."""
    print("🚀 LLMFlow Transport Layer Debug\n")
    
    # Check dependencies first
    if not check_missing_dependencies():
        print("\n❌ Missing dependencies. Please install them first.")
        return False
    
    # Test imports
    if not debug_imports():
        print("\n❌ Import issues detected. Check your Python environment.")
        return False
    
    # Test UDP
    udp_success = await test_udp_basic()
    
    # Test TCP
    tcp_success = await test_tcp_basic()
    
    # Test message sending
    messaging_success = await test_message_sending()
    
    # Summary
    print("\n📊 Debug Summary:")
    print(f"  UDP Basic: {'✅' if udp_success else '❌'}")
    print(f"  TCP Basic: {'✅' if tcp_success else '❌'}")
    print(f"  Messaging: {'✅' if messaging_success else '❌'}")
    
    overall_success = udp_success and tcp_success and messaging_success
    print(f"\nOverall: {'✅ All tests passed!' if overall_success else '❌ Some tests failed'}")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
