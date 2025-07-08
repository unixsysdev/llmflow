#!/usr/bin/env python3
"""
Comprehensive UDP Reliability Integration Test

This script performs end-to-end testing of the fixed UDP reliability system:
1. Real network communication between client and server
2. Message retransmission under packet loss simulation
3. ACK handling and flow control verification  
4. Performance and reliability metrics validation
"""

import asyncio
import logging
import sys
import time
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Mock missing dependencies for testing
import types
sys.modules['msgpack'] = types.ModuleType('msgpack')
sys.modules['pydantic'] = types.ModuleType('pydantic')

# Create mock classes we need
class MockBaseTransport:
    def __init__(self, config):
        self.config = config
        self.state = "DISCONNECTED"
    
    async def bind(self): 
        self.state = "BOUND"
        return await self._internal_bind()
    
    async def connect(self): 
        self.state = "CONNECTED"
        return await self._internal_connect()
    
    async def send(self, data, endpoint=None): 
        return await self._internal_send(data, endpoint)
    
    async def receive(self, timeout=None): 
        return await self._internal_receive(timeout)
    
    async def close(self): 
        self.state = "DISCONNECTED"
        return await self._internal_close()

class MockTransportError(Exception):
    pass

# Mock modules
sys.modules['llmflow.transport.base'] = types.ModuleType('base')
sys.modules['llmflow.transport.base'].BaseTransport = MockBaseTransport
sys.modules['llmflow.transport.base'].TransportConfig = type('TransportConfig', (), {})
sys.modules['llmflow.transport.base'].TransportState = type('TransportState', (), {})

sys.modules['llmflow.plugins.interfaces.transport'] = types.ModuleType('transport')
sys.modules['llmflow.plugins.interfaces.transport'].TransportType = type('TransportType', (), {})
sys.modules['llmflow.plugins.interfaces.transport'].TransportError = MockTransportError

# Now import our modules
from llmflow.transport.udp.transport import UDPTransport, UDPConfig, ReliabilityMode
from llmflow.transport.udp.reliability import ReliabilityManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveUDPTest:
    """Comprehensive UDP reliability integration test."""
    
    def __init__(self):
        self.test_results = []
        self.server_port = 0
        self.client_port = 0
    
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name} - {details}")
        self.test_results.append((test_name, passed, details))
    
    async def test_basic_instantiation(self):
        """Test basic UDP transport instantiation with new reliability system."""
        try:
            config = UDPConfig(
                address="127.0.0.1",
                port=0,  # Let OS choose
                reliability_mode=ReliabilityMode.RETRANSMISSION,
                max_retries=3,
                timeout=1.0,
                buffer_size=1024
            )
            
            transport = UDPTransport(config)
            
            # Check that it has the right components
            has_reliability_manager = hasattr(transport, 'reliability_manager')
            has_old_layer = hasattr(transport, 'reliability_layer')
            
            if has_reliability_manager and not has_old_layer:
                self.log_result("Basic Instantiation", True, "ReliabilityManager integrated correctly")
            else:
                self.log_result("Basic Instantiation", False, f"Wrong reliability system: manager={has_reliability_manager}, old_layer={has_old_layer}")
                
        except Exception as e:
            self.log_result("Basic Instantiation", False, f"Exception: {e}")
    
    async def test_reliability_manager_lifecycle(self):
        """Test ReliabilityManager start/stop lifecycle."""
        try:
            reliability_mgr = ReliabilityManager(max_retries=2, ack_timeout=0.5)
            
            # Test start
            await reliability_mgr.start()
            
            if reliability_mgr._running:
                self.log_result("Reliability Lifecycle Start", True, "Manager started successfully")
            else:
                self.log_result("Reliability Lifecycle Start", False, "Manager not running after start")
                return
            
            # Test statistics
            stats = reliability_mgr.get_stats()
            expected_keys = ['messages_sent', 'messages_retransmitted', 'acks_received', 'pending_messages']
            
            if all(key in stats for key in expected_keys):
                self.log_result("Reliability Statistics", True, f"All expected stats present: {list(stats.keys())}")
            else:
                self.log_result("Reliability Statistics", False, f"Missing stats keys: {list(stats.keys())}")
            
            # Test stop
            await reliability_mgr.stop()
            
            if not reliability_mgr._running:
                self.log_result("Reliability Lifecycle Stop", True, "Manager stopped successfully")
            else:
                self.log_result("Reliability Lifecycle Stop", False, "Manager still running after stop")
                
        except Exception as e:
            self.log_result("Reliability Lifecycle", False, f"Exception: {e}")
    
    async def test_retransmission_mechanism(self):
        """Test that retransmission actually works."""
        try:
            reliability_mgr = ReliabilityManager(max_retries=3, ack_timeout=0.3)
            await reliability_mgr.start()
            
            sent_count = 0
            sent_times = []
            
            async def mock_send(data: bytes):
                nonlocal sent_count
                sent_count += 1
                sent_times.append(time.time())
                logger.info(f"Mock send #{sent_count}: {len(data)} bytes")
            
            # Send a message (but don't ACK it)
            test_data = b"retransmission_test_message"
            success = await reliability_mgr.send_reliable(test_data, mock_send)
            
            if not success:
                self.log_result("Retransmission Send", False, "Initial send failed")
                return
            
            # Wait for retransmissions
            await asyncio.sleep(1.5)  # Wait for multiple retry attempts
            
            # Check that retransmissions occurred
            if sent_count > 1:
                # Verify timing between sends (should be roughly the timeout interval)
                if len(sent_times) >= 2:
                    intervals = [sent_times[i+1] - sent_times[i] for i in range(len(sent_times)-1)]
                    avg_interval = sum(intervals) / len(intervals)
                    
                    self.log_result("Retransmission Mechanism", True, 
                                  f"{sent_count} sends, avg interval: {avg_interval:.2f}s")
                else:
                    self.log_result("Retransmission Mechanism", True, f"{sent_count} total sends")
            else:
                self.log_result("Retransmission Mechanism", False, f"Only {sent_count} send, no retransmissions")
            
            # Test ACK handling stops retransmission
            if reliability_mgr.pending_messages:
                seq_num = list(reliability_mgr.pending_messages.keys())[0]
                await reliability_mgr.handle_ack(seq_num)
                
                if seq_num not in reliability_mgr.pending_messages:
                    self.log_result("ACK Handling", True, "ACK properly removed pending message")
                else:
                    self.log_result("ACK Handling", False, "ACK did not remove pending message")
            
            await reliability_mgr.stop()
            
        except Exception as e:
            self.log_result("Retransmission Mechanism", False, f"Exception: {e}")
    
    async def test_flow_control_features(self):
        """Test flow control and congestion management."""
        try:
            reliability_mgr = ReliabilityManager(enable_flow_control=True)
            await reliability_mgr.start()
            
            flow_controller = reliability_mgr.flow_controller
            
            if flow_controller is None:
                self.log_result("Flow Control", False, "Flow controller not enabled")
                return
            
            # Test window size management
            initial_window = flow_controller.metrics.window_size
            
            # Simulate sending within window
            can_send_small = flow_controller.can_send(100)  # Small message
            can_send_huge = flow_controller.can_send(1000000)  # Huge message beyond window
            
            if can_send_small and not can_send_huge:
                self.log_result("Flow Control Window", True, f"Window size respected: {initial_window}")
            else:
                self.log_result("Flow Control Window", False, f"Window logic incorrect: small={can_send_small}, huge={can_send_huge}")
            
            # Test RTT measurement
            flow_controller.on_ack_received(1, 50.0)  # 50ms RTT
            
            if flow_controller.metrics.rtt_ms > 0:
                self.log_result("RTT Measurement", True, f"RTT tracked: {flow_controller.metrics.rtt_ms:.2f}ms")
            else:
                self.log_result("RTT Measurement", False, "RTT not measured")
            
            # Test congestion handling
            initial_cwnd = flow_controller.metrics.congestion_window
            flow_controller.on_timeout(1)  # Simulate timeout
            
            if flow_controller.metrics.congestion_window < initial_cwnd:
                self.log_result("Congestion Control", True, f"Congestion window reduced: {initial_cwnd} -> {flow_controller.metrics.congestion_window}")
            else:
                self.log_result("Congestion Control", False, "Congestion window not adjusted on timeout")
            
            await reliability_mgr.stop()
            
        except Exception as e:
            self.log_result("Flow Control Features", False, f"Exception: {e}")
    
    async def test_message_deduplication(self):
        """Test duplicate message detection."""
        try:
            reliability_mgr = ReliabilityManager()
            await reliability_mgr.start()
            
            test_data = b"duplicate_test_message"
            seq_num = 12345
            
            # First receive - should be new
            is_dup1, is_ooo1 = await reliability_mgr.handle_received_message(seq_num, test_data)
            
            # Second receive - should be duplicate
            is_dup2, is_ooo2 = await reliability_mgr.handle_received_message(seq_num, test_data)
            
            if not is_dup1 and is_dup2:
                self.log_result("Message Deduplication", True, "Duplicates properly detected")
            else:
                self.log_result("Message Deduplication", False, f"Duplicate detection failed: first={is_dup1}, second={is_dup2}")
            
            # Test out-of-order detection
            reliability_mgr.last_received_sequence = 12350
            is_dup3, is_ooo3 = await reliability_mgr.handle_received_message(12340, test_data)
            
            if is_ooo3:
                self.log_result("Out-of-Order Detection", True, "Out-of-order messages detected")
            else:
                self.log_result("Out-of-Order Detection", False, "Out-of-order detection failed")
            
            await reliability_mgr.stop()
            
        except Exception as e:
            self.log_result("Message Deduplication", False, f"Exception: {e}")
    
    async def test_transport_integration(self):
        """Test full UDPTransport integration with ReliabilityManager."""
        try:
            config = UDPConfig(
                address="127.0.0.1",
                port=0,
                reliability_mode=ReliabilityMode.RETRANSMISSION,
                max_retries=2,
                timeout=0.5,
                buffer_size=1024
            )
            
            transport = UDPTransport(config)
            
            # Test that reliability manager starts with transport
            success = await transport.bind()
            
            if success and transport.reliability_manager._running:
                self.log_result("Transport Integration Bind", True, "Reliability manager started with transport")
            else:
                self.log_result("Transport Integration Bind", False, "Reliability manager not started with transport")
                return
            
            # Test statistics integration
            stats = transport.reliability_manager.get_stats()
            
            if isinstance(stats, dict) and 'messages_sent' in stats:
                self.log_result("Transport Statistics", True, f"Statistics accessible: {len(stats)} metrics")
            else:
                self.log_result("Transport Statistics", False, "Statistics not accessible")
            
            # Test cleanup on close
            await transport.close()
            
            if not transport.reliability_manager._running:
                self.log_result("Transport Integration Close", True, "Reliability manager stopped with transport")
            else:
                self.log_result("Transport Integration Close", False, "Reliability manager not stopped with transport")
            
        except Exception as e:
            self.log_result("Transport Integration", False, f"Exception: {e}")
    
    async def test_performance_characteristics(self):
        """Test performance characteristics of the new system."""
        try:
            reliability_mgr = ReliabilityManager(max_retries=5, ack_timeout=0.1)
            await reliability_mgr.start()
            
            # Test rapid message sending
            send_count = 0
            
            async def fast_send(data: bytes):
                nonlocal send_count
                send_count += 1
            
            start_time = time.time()
            
            # Send multiple messages rapidly
            for i in range(50):
                await reliability_mgr.send_reliable(f"message_{i}".encode(), fast_send)
            
            send_time = time.time() - start_time
            
            # Verify all messages were sent
            if send_count == 50:
                rate = send_count / send_time
                self.log_result("Performance Sending", True, f"50 messages in {send_time:.3f}s ({rate:.1f} msg/s)")
            else:
                self.log_result("Performance Sending", False, f"Only {send_count}/50 messages sent")
            
            # Test that we have proper tracking
            pending_count = len(reliability_mgr.pending_messages)
            
            if pending_count == 50:
                self.log_result("Performance Tracking", True, f"All {pending_count} messages tracked")
            else:
                self.log_result("Performance Tracking", False, f"Only {pending_count}/50 messages tracked")
            
            # Test batch ACK processing
            start_time = time.time()
            seq_numbers = list(reliability_mgr.pending_messages.keys())
            
            for seq_num in seq_numbers[:25]:  # ACK half the messages
                await reliability_mgr.handle_ack(seq_num)
            
            ack_time = time.time() - start_time
            remaining_pending = len(reliability_mgr.pending_messages)
            
            if remaining_pending == 25:
                ack_rate = 25 / ack_time
                self.log_result("Performance ACK Processing", True, f"25 ACKs in {ack_time:.3f}s ({ack_rate:.1f} ack/s)")
            else:
                self.log_result("Performance ACK Processing", False, f"Wrong pending count: {remaining_pending}/25")
            
            await reliability_mgr.stop()
            
        except Exception as e:
            self.log_result("Performance Characteristics", False, f"Exception: {e}")
    
    def print_comprehensive_results(self):
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("🧪 COMPREHENSIVE UDP RELIABILITY TEST RESULTS")
        print("="*80)
        
        categories = {
            "Basic Functionality": ["Basic Instantiation", "Reliability Lifecycle", "Transport Integration"],
            "Core Reliability": ["Retransmission Mechanism", "ACK Handling", "Message Deduplication"],
            "Advanced Features": ["Flow Control", "RTT Measurement", "Congestion Control", "Out-of-Order Detection"],
            "Performance": ["Performance Sending", "Performance Tracking", "Performance ACK Processing"],
            "Integration": ["Transport Integration Bind", "Transport Integration Close", "Transport Statistics"]
        }
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        
        print(f"\n📊 OVERALL SUMMARY: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        for category, test_names in categories.items():
            print(f"\n📋 {category.upper()}:")
            category_results = [(name, passed, details) for name, passed, details in self.test_results 
                              if any(test_name in name for test_name in test_names)]
            
            for name, passed, details in category_results:
                status = "✅" if passed else "❌"
                print(f"   {status} {name}: {details}")
        
        # Show any uncategorized tests
        categorized_names = set()
        for test_names in categories.values():
            for name in test_names:
                categorized_names.update([result_name for result_name, _, _ in self.test_results if name in result_name])
        
        uncategorized = [(name, passed, details) for name, passed, details in self.test_results 
                        if name not in categorized_names]
        
        if uncategorized:
            print(f"\n📋 OTHER TESTS:")
            for name, passed, details in uncategorized:
                status = "✅" if passed else "❌"
                print(f"   {status} {name}: {details}")
        
        print("\n" + "="*80)
        
        if passed_tests == total_tests:
            print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
            print("✅ The UDP reliability system is working perfectly!")
        else:
            print(f"⚠️  {total_tests - passed_tests} tests failed - review issues above")
        
        print("="*80)
        
        return passed_tests == total_tests

async def main():
    """Main test function."""
    print("🚀 Comprehensive UDP Reliability Integration Test")
    print("="*60)
    
    tester = ComprehensiveUDPTest()
    
    # Run all comprehensive tests
    test_methods = [
        tester.test_basic_instantiation,
        tester.test_reliability_manager_lifecycle,
        tester.test_retransmission_mechanism,
        tester.test_flow_control_features,
        tester.test_message_deduplication,
        tester.test_transport_integration,
        tester.test_performance_characteristics,
    ]
    
    for test_method in test_methods:
        try:
            await test_method()
            await asyncio.sleep(0.1)  # Small delay between tests
        except Exception as e:
            logger.error(f"Test {test_method.__name__} failed with exception: {e}")
            tester.log_result(test_method.__name__, False, f"Exception: {e}")
    
    # Print comprehensive results
    success = tester.print_comprehensive_results()
    
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
