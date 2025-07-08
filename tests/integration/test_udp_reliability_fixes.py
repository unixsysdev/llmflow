#!/usr/bin/env python3
"""
Test the fixed UDP reliability implementation.

This script tests the three critical fixes:
1. Retransmission is properly implemented
2. ACK forwarding works correctly  
3. Single unified message tracking
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llmflow.transport.udp.transport import UDPTransport, UDPConfig, ReliabilityMode
from llmflow.transport.udp.reliability import ReliabilityManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UDPReliabilityFixTester:
    """Test the UDP reliability fixes."""
    
    def __init__(self):
        self.test_results = []
    
    async def test_retransmission_fix(self):
        """Test that retransmission is properly implemented."""
        logger.info("🧪 Testing Retransmission Fix...")
        
        # Create reliability manager with short timeout for testing
        reliability_mgr = ReliabilityManager(max_retries=2, ack_timeout=0.5)
        await reliability_mgr.start()
        
        sent_messages = []
        
        async def mock_send(data: bytes):
            sent_messages.append(time.time())
            logger.info(f"📤 Mock send called - total calls: {len(sent_messages)}")
        
        try:
            # Send a message (but don't send ACK)
            await reliability_mgr.send_reliable(b"test_retransmission", mock_send)
            
            # Wait for retransmissions
            await asyncio.sleep(2.0)
            
            # Should have original send + retransmissions
            if len(sent_messages) > 1:
                logger.info(f"✅ RETRANSMISSION WORKS: {len(sent_messages)} sends (original + retries)")
                self.test_results.append(("Retransmission", True, f"{len(sent_messages)} total sends"))
            else:
                logger.error(f"❌ RETRANSMISSION FAILED: Only {len(sent_messages)} sends")
                self.test_results.append(("Retransmission", False, f"Only {len(sent_messages)} sends"))
                
        finally:
            await reliability_mgr.stop()
    
    async def test_ack_integration_fix(self):
        """Test that ACKs properly integrate with reliability manager."""
        logger.info("🧪 Testing ACK Integration Fix...")
        
        config = UDPConfig(
            address="127.0.0.1",
            port=0,  # Let OS choose port
            reliability_mode=ReliabilityMode.RETRANSMISSION,
            max_retries=3,
            timeout=1.0
        )
        
        transport = UDPTransport(config)
        
        try:
            # Start transport (this starts reliability manager)
            await transport.bind()
            
            # Check that transport uses the reliability manager
            if hasattr(transport, 'reliability_manager'):
                reliability_mgr = transport.reliability_manager
                
                # Send a message to create pending message
                test_data = b"test_ack_integration"
                result = await transport.send(test_data, ("127.0.0.1", 9999))
                
                if result:
                    # Get a sequence number from pending messages
                    if reliability_mgr.pending_messages:
                        seq_num = list(reliability_mgr.pending_messages.keys())[0]
                        logger.info(f"📝 Created pending message {seq_num}")
                        
                        # Simulate ACK received
                        await reliability_mgr.handle_ack(seq_num)
                        
                        # Check if message was removed from pending
                        if seq_num not in reliability_mgr.pending_messages:
                            logger.info("✅ ACK INTEGRATION WORKS: Message removed from pending")
                            self.test_results.append(("ACK Integration", True, "Message properly acknowledged"))
                        else:
                            logger.error("❌ ACK INTEGRATION FAILED: Message still pending")
                            self.test_results.append(("ACK Integration", False, "Message not acknowledged"))
                    else:
                        logger.error("❌ ACK INTEGRATION FAILED: No pending messages created")
                        self.test_results.append(("ACK Integration", False, "No pending messages"))
                else:
                    logger.error("❌ ACK INTEGRATION FAILED: Send failed")
                    self.test_results.append(("ACK Integration", False, "Send failed"))
            else:
                logger.error("❌ ACK INTEGRATION FAILED: No reliability_manager attribute")
                self.test_results.append(("ACK Integration", False, "No reliability manager"))
                
        finally:
            await transport.close()
    
    async def test_unified_tracking_fix(self):
        """Test that message tracking is unified in reliability manager."""
        logger.info("🧪 Testing Unified Message Tracking...")
        
        config = UDPConfig(
            address="127.0.0.1", 
            port=0,
            reliability_mode=ReliabilityMode.RETRANSMISSION
        )
        
        transport = UDPTransport(config)
        
        try:
            await transport.bind()
            
            # Check for unified tracking (no dual systems)
            has_reliability_manager = hasattr(transport, 'reliability_manager')
            has_old_reliability_layer = hasattr(transport, 'reliability_layer')
            
            if has_reliability_manager and not has_old_reliability_layer:
                logger.info("✅ UNIFIED TRACKING: Only reliability_manager exists")
                self.test_results.append(("Unified Tracking", True, "Single tracking system"))
            elif has_reliability_manager and has_old_reliability_layer:
                logger.warning("⚠️  PARTIAL FIX: Both systems exist")
                self.test_results.append(("Unified Tracking", False, "Dual systems still exist"))
            else:
                logger.error("❌ UNIFIED TRACKING FAILED: Wrong configuration")
                self.test_results.append(("Unified Tracking", False, "Wrong configuration"))
                
        finally:
            await transport.close()
    
    def print_results(self):
        """Print test results."""
        print("\n" + "="*60)
        print("🔍 UDP RELIABILITY FIX TEST RESULTS")
        print("="*60)
        
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        total_tests = len(self.test_results)
        
        for test_name, passed, details in self.test_results:
            status = "✅ PASS" if passed else "❌ FAIL" 
            print(f"{status}: {test_name} - {details}")
        
        print(f"\n📊 SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL FIXES WORKING CORRECTLY!")
            return True
        else:
            print("⚠️  SOME ISSUES REMAIN")
            return False

async def main():
    """Main test function."""
    print("🧪 LLMFlow UDP Reliability Fix Tester")
    print("="*50)
    
    tester = UDPReliabilityFixTester()
    
    # Run all tests
    await tester.test_retransmission_fix()
    await asyncio.sleep(0.5)
    
    await tester.test_ack_integration_fix() 
    await asyncio.sleep(0.5)
    
    await tester.test_unified_tracking_fix()
    
    # Print results
    success = tester.print_results()
    
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
        sys.exit(1)
