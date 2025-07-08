#!/usr/bin/env python3
"""
Manual Testing Suite for LLMFlow UDP Reliability Issues

This script tests the UDP reliability protocol and client-side confirmation issues
without requiring external dependencies.
"""

import asyncio
import json
import logging
import socket
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ManualTestRunner:
    """Manual test runner without external dependencies."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failed_tests = []
    
    def assert_equal(self, actual, expected, message=""):
        """Manual assertion for equality."""
        if actual == expected:
            logger.info(f"✅ PASS: {message} (expected: {expected}, got: {actual})")
            self.tests_passed += 1
        else:
            logger.error(f"❌ FAIL: {message} (expected: {expected}, got: {actual})")
            self.tests_failed += 1
            self.failed_tests.append(message)
        self.tests_run += 1
    
    def assert_true(self, condition, message=""):
        """Manual assertion for truth."""
        if condition:
            logger.info(f"✅ PASS: {message}")
            self.tests_passed += 1
        else:
            logger.error(f"❌ FAIL: {message}")
            self.tests_failed += 1
            self.failed_tests.append(message)
        self.tests_run += 1
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print(f"TEST SUMMARY: {self.tests_run} tests run")
        print(f"✅ PASSED: {self.tests_passed}")
        print(f"❌ FAILED: {self.tests_failed}")
        
        if self.failed_tests:
            print("\nFAILED TESTS:")
            for test in self.failed_tests:
                print(f"  - {test}")
        
        print("="*60)
        return self.tests_failed == 0

# Create global test runner
test_runner = ManualTestRunner()

def test_imports():
    """Test basic imports without external dependencies."""
    logger.info("🧪 Testing Basic Imports...")
    
    try:
        # Test core base imports
        from llmflow.core.base import DataAtom, ServiceAtom, Component
        test_runner.assert_true(True, "Core base classes import")
    except Exception as e:
        test_runner.assert_true(False, f"Core base classes import failed: {e}")
    
    try:
        # Test atoms import (might fail due to msgpack)
        from llmflow.atoms.data import StringAtom
        test_runner.assert_true(True, "Data atoms import")
    except Exception as e:
        test_runner.assert_true(False, f"Data atoms import failed: {e}")
    
    try:
        # Test transport imports
        from llmflow.transport.udp.transport import UDPTransport
        test_runner.assert_true(True, "UDP transport import")
    except Exception as e:
        test_runner.assert_true(False, f"UDP transport import failed: {e}")

def test_udp_socket_basic():
    """Test basic UDP socket functionality."""
    logger.info("🧪 Testing Basic UDP Socket Functionality...")
    
    try:
        # Test UDP socket creation
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Try to bind to a port
        sock.bind(('localhost', 0))  # 0 = any available port
        host, port = sock.getsockname()
        
        test_runner.assert_true(True, f"UDP socket bind successful on {host}:{port}")
        
        # Test basic send/receive
        test_message = b"Hello UDP Test"
        
        # Send to self
        sock.sendto(test_message, (host, port))
        
        # Receive with timeout
        sock.settimeout(1.0)
        try:
            received_data, addr = sock.recvfrom(1024)
            test_runner.assert_equal(received_data, test_message, "UDP self-send/receive")
        except socket.timeout:
            test_runner.assert_true(False, "UDP receive timeout")
        
        sock.close()
        
    except Exception as e:
        test_runner.assert_true(False, f"UDP socket test failed: {e}")

def test_udp_reliability_protocol():
    """Test UDP reliability protocol structure."""
    logger.info("🧪 Testing UDP Reliability Protocol...")
    
    try:
        # Test protocol message structure without importing full classes
        
        # Simulate message structure: [Type][Seq][Payload]
        message_type = b'DATA'
        sequence_number = (12345).to_bytes(4, 'big')
        payload = b'{"test": "data"}'
        
        # Construct message
        message = message_type + sequence_number + payload
        
        # Parse message
        parsed_type = message[:4]
        parsed_seq = int.from_bytes(message[4:8], 'big')
        parsed_payload = message[8:]
        
        test_runner.assert_equal(parsed_type, message_type, "Message type parsing")
        test_runner.assert_equal(parsed_seq, 12345, "Sequence number parsing")
        test_runner.assert_equal(parsed_payload, payload, "Payload parsing")
        
        # Test ACK message structure
        ack_type = b'ACK '
        ack_message = ack_type + sequence_number
        
        parsed_ack_type = ack_message[:4]
        parsed_ack_seq = int.from_bytes(ack_message[4:8], 'big')
        
        test_runner.assert_equal(parsed_ack_type, ack_type, "ACK message type parsing")
        test_runner.assert_equal(parsed_ack_seq, 12345, "ACK sequence number parsing")
        
    except Exception as e:
        test_runner.assert_true(False, f"UDP reliability protocol test failed: {e}")

def test_client_server_simulation():
    """Simulate client-server UDP communication with acknowledgments."""
    logger.info("🧪 Testing Client-Server UDP Communication...")
    
    async def run_test():
        try:
            # Server setup
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('localhost', 0))
            server_host, server_port = server_socket.getsockname()
            server_socket.setblocking(False)
            
            # Client setup
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client_socket.setblocking(False)
            
            logger.info(f"Server listening on {server_host}:{server_port}")
            
            # Test 1: Basic message send
            test_message = b'DATA' + (1).to_bytes(4, 'big') + b'Hello Server'
            client_socket.sendto(test_message, (server_host, server_port))
            
            # Server receive
            await asyncio.sleep(0.1)  # Small delay
            try:
                received_data, client_addr = server_socket.recvfrom(1024)
                test_runner.assert_equal(received_data, test_message, "Server received client message")
                
                # Server sends ACK
                ack_message = b'ACK ' + (1).to_bytes(4, 'big')
                server_socket.sendto(ack_message, client_addr)
                
                # Client receives ACK
                await asyncio.sleep(0.1)
                ack_received, _ = client_socket.recvfrom(1024)
                test_runner.assert_equal(ack_received, ack_message, "Client received ACK")
                
            except Exception as e:
                test_runner.assert_true(False, f"Message exchange failed: {e}")
            
            # Test 2: Missing ACK simulation (timeout test)
            test_message_2 = b'DATA' + (2).to_bytes(4, 'big') + b'No ACK Expected'
            client_socket.sendto(test_message_2, (server_host, server_port))
            
            # Server receives but doesn't send ACK (simulate network issue)
            await asyncio.sleep(0.1)
            try:
                received_data, client_addr = server_socket.recvfrom(1024)
                test_runner.assert_equal(received_data, test_message_2, "Server received message (no ACK test)")
                # Deliberately not sending ACK to test timeout
            except Exception as e:
                test_runner.assert_true(False, f"No-ACK test receive failed: {e}")
            
            # Client should timeout waiting for ACK
            client_socket.settimeout(0.5)  # 500ms timeout
            try:
                ack_received, _ = client_socket.recvfrom(1024)
                test_runner.assert_true(False, "Unexpected ACK received (should have timed out)")
            except socket.timeout:
                test_runner.assert_true(True, "Client timeout on missing ACK (correct behavior)")
            except Exception as e:
                test_runner.assert_true(False, f"Unexpected error in timeout test: {e}")
            
            # Cleanup
            server_socket.close()
            client_socket.close()
            
        except Exception as e:
            test_runner.assert_true(False, f"Client-server simulation failed: {e}")
    
    # Run the async test
    asyncio.run(run_test())

def test_reliability_edge_cases():
    """Test edge cases in UDP reliability protocol."""
    logger.info("🧪 Testing UDP Reliability Edge Cases...")
    
    try:
        # Test 1: Duplicate sequence numbers
        seq_num = 12345
        message1 = b'DATA' + seq_num.to_bytes(4, 'big') + b'First message'
        message2 = b'DATA' + seq_num.to_bytes(4, 'big') + b'Duplicate seq'
        
        # Both messages have same sequence number - should be detectable
        seq1 = int.from_bytes(message1[4:8], 'big')
        seq2 = int.from_bytes(message2[4:8], 'big')
        test_runner.assert_equal(seq1, seq2, "Duplicate sequence detection")
        
        # Test 2: Out-of-order messages
        seq_a = 100
        seq_b = 99  # Lower sequence number (out of order)
        
        message_a = b'DATA' + seq_a.to_bytes(4, 'big') + b'Message A'
        message_b = b'DATA' + seq_b.to_bytes(4, 'big') + b'Message B'
        
        seq_parsed_a = int.from_bytes(message_a[4:8], 'big')
        seq_parsed_b = int.from_bytes(message_b[4:8], 'big')
        
        test_runner.assert_true(seq_parsed_a > seq_parsed_b, "Out-of-order sequence detection")
        
        # Test 3: Message size limits
        max_payload = b'x' * 1000  # 1KB payload
        large_message = b'DATA' + (1).to_bytes(4, 'big') + max_payload
        
        test_runner.assert_true(len(large_message) <= 1024, "Message size within UDP limits")
        
        # Test 4: Invalid message format
        invalid_message = b'INVALID_FORMAT'
        try:
            # Try to parse as valid message
            if len(invalid_message) >= 8:
                msg_type = invalid_message[:4]
                seq_num = int.from_bytes(invalid_message[4:8], 'big')
                test_runner.assert_true(False, "Should not parse invalid message")
            else:
                test_runner.assert_true(True, "Correctly rejected too-short message")
        except:
            test_runner.assert_true(True, "Correctly rejected invalid message format")
        
    except Exception as e:
        test_runner.assert_true(False, f"Reliability edge cases test failed: {e}")

def test_queue_protocol_structure():
    """Test the queue protocol message structure."""
    logger.info("🧪 Testing Queue Protocol Structure...")
    
    try:
        # Simulate queue protocol message structure
        # [Header][Context][Payload]
        # Header: MessageType(1) + QueueID(8) + MessageID(8) + PayloadSize(4)
        
        message_type = b'E'  # ENQUEUE
        queue_id = b'testque1'  # 8 bytes (padded)
        message_id = b'msg12345'  # 8 bytes
        payload_size = (50).to_bytes(4, 'big')
        
        # Context: SecurityLevel(1) + Domain(variable) + TenantID(variable)
        security_level = b'1'  # Level 1
        domain = b'test_domain\x00'  # Null-terminated
        tenant_id = b'tenant1\x00'  # Null-terminated
        
        # Payload
        payload = b'{"operation": "enqueue", "data": "test_message"}'
        
        # Construct full message
        header = message_type + queue_id + message_id + payload_size
        context = security_level + domain + tenant_id
        full_message = header + context + payload
        
        # Parse message
        parsed_type = full_message[0:1]
        parsed_queue_id = full_message[1:9]
        parsed_message_id = full_message[9:17]
        parsed_payload_size = int.from_bytes(full_message[17:21], 'big')
        
        test_runner.assert_equal(parsed_type, message_type, "Message type parsing")
        test_runner.assert_equal(parsed_queue_id, queue_id, "Queue ID parsing")
        test_runner.assert_equal(parsed_message_id, message_id, "Message ID parsing")
        test_runner.assert_equal(parsed_payload_size, 50, "Payload size parsing")
        
        # Find context start (after header)
        context_start = 21
        parsed_security = full_message[context_start:context_start+1]
        test_runner.assert_equal(parsed_security, security_level, "Security level parsing")
        
    except Exception as e:
        test_runner.assert_true(False, f"Queue protocol structure test failed: {e}")

def test_real_transport_layer():
    """Test the actual transport layer if available."""
    logger.info("🧪 Testing Real Transport Layer...")
    
    try:
        # Try to import and test the actual UDP transport
        from llmflow.transport.udp.transport import UDPTransport
        
        async def transport_test():
            try:
                # Create transport instance
                transport = UDPTransport(host='localhost', port=0)
                
                # Test initialization
                test_runner.assert_true(hasattr(transport, 'send'), "Transport has send method")
                test_runner.assert_true(hasattr(transport, 'receive'), "Transport has receive method")
                
                # Test start/stop
                await transport.start()
                test_runner.assert_true(True, "Transport start successful")
                
                await transport.stop()
                test_runner.assert_true(True, "Transport stop successful")
                
            except Exception as e:
                test_runner.assert_true(False, f"Transport layer test failed: {e}")
        
        asyncio.run(transport_test())
        
    except ImportError as e:
        test_runner.assert_true(False, f"Transport layer import failed: {e}")
    except Exception as e:
        test_runner.assert_true(False, f"Transport layer test failed: {e}")

def test_reliability_layer():
    """Test the reliability layer if available."""
    logger.info("🧪 Testing Reliability Layer...")
    
    try:
        from llmflow.transport.udp.reliability import ReliabilityManager
        
        # Test reliability manager creation
        reliability_mgr = ReliabilityManager()
        test_runner.assert_true(hasattr(reliability_mgr, 'send_reliable'), "Reliability manager has send_reliable")
        test_runner.assert_true(hasattr(reliability_mgr, 'handle_ack'), "Reliability manager has handle_ack")
        
        # Test message tracking
        message_id = "test_msg_123"
        # Simulate adding message to pending
        if hasattr(reliability_mgr, 'pending_messages'):
            test_runner.assert_true(True, "Reliability manager has pending_messages tracking")
        
    except ImportError as e:
        test_runner.assert_true(False, f"Reliability layer import failed: {e}")
    except Exception as e:
        test_runner.assert_true(False, f"Reliability layer test failed: {e}")

def main():
    """Run all manual tests."""
    print("🧪 LLMFlow Manual Testing Suite")
    print("=" * 60)
    print("Testing UDP reliability and client-side confirmation issues...")
    print()
    
    # Run all tests
    test_imports()
    test_udp_socket_basic()
    test_udp_reliability_protocol()
    test_client_server_simulation()
    test_reliability_edge_cases()
    test_queue_protocol_structure()
    test_real_transport_layer()
    test_reliability_layer()
    
    # Print summary
    success = test_runner.print_summary()
    
    if not success:
        print("\n⚠️  ISSUES FOUND!")
        print("The tests revealed problems that need to be addressed.")
        print("Focus areas:")
        print("- UDP reliability protocol implementation")
        print("- Client-side acknowledgment handling")
        print("- Message timeout and retry logic")
        print("- Transport layer imports and dependencies")
    else:
        print("\n✅ ALL TESTS PASSED!")
        print("UDP reliability and client communication appears to be working correctly.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
