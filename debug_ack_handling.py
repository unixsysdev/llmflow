"""
Debug ACK Handling

Test to specifically debug the ACK handling in the reliability layer.
"""

import asyncio
import sys
sys.path.insert(0, '.')

async def test_ack_handling():
    """Test ACK sending and receiving."""
    print("🔍 Testing ACK handling...")
    
    try:
        from llmflow.transport.udp.transport import UDPTransport, UDPConfig, ReliabilityMode
        
        # Enable more detailed logging
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        # Create server and client with acknowledgment mode
        server_config = UDPConfig(
            address="127.0.0.1", 
            port=18200, 
            reliability_mode=ReliabilityMode.ACKNOWLEDGMENT,
            ack_timeout=0.5,
            max_retries=2
        )
        server_transport = UDPTransport(server_config)
        
        client_config = UDPConfig(
            address="127.0.0.1", 
            port=18200,
            reliability_mode=ReliabilityMode.ACKNOWLEDGMENT,
            ack_timeout=0.5,
            max_retries=2
        )
        client_transport = UDPTransport(client_config)
        
        try:
            # Bind server
            server_success = await server_transport.bind("127.0.0.1", 18200)
            print(f"Server bind: {server_success}")
            
            # Connect client
            client_success = await client_transport.connect("127.0.0.1", 18200)
            print(f"Client connect: {client_success}")
            
            if server_success and client_success:
                # Start receiving task for server
                async def server_receive_task():
                    print("Server: Starting receive loop...")
                    for i in range(10):  # Try to receive for a reasonable time
                        try:
                            result = await server_transport.receive(timeout=0.5)
                            if result:
                                data, sender = result
                                print(f"Server: Received data: {data} from {sender}")
                                return data
                        except Exception as e:
                            print(f"Server: Receive error: {e}")
                    print("Server: No data received in receive loop")
                    return None
                
                # Start the server receive task
                receive_task = asyncio.create_task(server_receive_task())
                
                # Wait a moment for server to be ready
                await asyncio.sleep(0.1)
                
                # Send message from client
                print("Client: Sending message...")
                test_message = b"ACK test message"
                send_success = await client_transport.send(test_message)
                print(f"Client: Send result: {send_success}")
                
                # Wait for receive task to complete
                received_data = await receive_task
                
                if received_data:
                    print(f"✓ Message successfully sent and received: {received_data}")
                    success = received_data == test_message
                    print(f"Message match: {success}")
                    return success
                else:
                    print("❌ No message received")
                    return False
            else:
                print("❌ Failed to set up server or client")
                return False
                
        finally:
            await server_transport.close()
            await client_transport.close()
            
    except Exception as e:
        print(f"❌ ACK test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main debug function."""
    print("🚀 ACK Handling Debug\n")
    
    success = await test_ack_handling()
    
    print(f"\nResult: {'✅ ACK handling works!' if success else '❌ ACK handling failed'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
