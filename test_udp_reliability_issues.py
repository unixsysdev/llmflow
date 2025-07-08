#!/usr/bin/env python3
"""
UDP Reliability Issues Test & Fix

This script identifies and tests the specific UDP reliability and client-side
confirmation issues in the LLMFlow transport layer.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UDPReliabilityTester:
    """Test UDP reliability issues."""
    
    def __init__(self):
        self.issues_found = []
        self.tests_run = 0
        self.tests_passed = 0
    
    def log_issue(self, issue_description):
        """Log an issue found."""
        self.issues_found.append(issue_description)
        logger.error(f"🐛 ISSUE FOUND: {issue_description}")
    
    def log_pass(self, test_description):
        """Log a test pass."""
        logger.info(f"✅ PASS: {test_description}")
        self.tests_passed += 1
    
    def log_fail(self, test_description, details=""):
        """Log a test failure."""
        logger.error(f"❌ FAIL: {test_description} - {details}")
    
    def test_reliability_layer_retransmission(self):
        """Test the retransmission implementation in reliability layer."""
        self.tests_run += 1
        
        try:
            # Read the reliability layer code to check for TODO
            reliability_file = Path("llmflow/transport/udp/reliability.py")
            if reliability_file.exists():
                content = reliability_file.read_text()
                
                # Check for the specific TODO that indicates missing retransmission
                if "TODO: Implement actual retransmission" in content:
                    self.log_issue("Retransmission is not implemented in ReliabilityManager._handle_timeout()")
                    return False
                else:
                    self.log_pass("Retransmission appears to be implemented")
                    return True
            else:
                self.log_fail("Reliability layer file not found")
                return False
                
        except Exception as e:
            self.log_fail("Could not test retransmission implementation", str(e))
            return False
    
    def test_ack_forwarding_integration(self):
        """Test ACK forwarding between transport and reliability layers."""
        self.tests_run += 1
        
        try:
            # Read transport layer to check ACK handling
            transport_file = Path("llmflow/transport/udp/transport.py")
            if transport_file.exists():
                content = transport_file.read_text()
                
                # Look for ACK handling that connects to reliability layer
                if "_handle_ack_message" in content:
                    # Check if ACK is forwarded to reliability layer
                    if "reliability_layer.handle_ack" in content:
                        self.log_pass("ACK forwarding to reliability layer found")
                        return True
                    else:
                        self.log_issue("ACK messages are handled in transport layer but not forwarded to reliability layer")
                        return False
                else:
                    self.log_fail("ACK handling not found in transport layer")
                    return False
            else:
                self.log_fail("Transport layer file not found")
                return False
                
        except Exception as e:
            self.log_fail("Could not test ACK forwarding", str(e))
            return False
    
    def test_pending_message_tracking(self):
        """Test consistency between transport and reliability layer message tracking."""
        self.tests_run += 1
        
        try:
            # Read both files to check for dual tracking
            transport_file = Path("llmflow/transport/udp/transport.py")
            reliability_file = Path("llmflow/transport/udp/reliability.py")
            
            if transport_file.exists() and reliability_file.exists():
                transport_content = transport_file.read_text()
                reliability_content = reliability_file.read_text()
                
                # Check for dual tracking systems
                transport_has_pending = "pending_acks" in transport_content
                reliability_has_pending = "pending_messages" in reliability_content
                
                if transport_has_pending and reliability_has_pending:
                    self.log_issue("Dual message tracking systems - transport has 'pending_acks' and reliability has 'pending_messages'")
                    return False
                elif reliability_has_pending:
                    self.log_pass("Single message tracking system in reliability layer")
                    return True
                else:
                    self.log_fail("No proper message tracking found")
                    return False
            else:
                self.log_fail("Transport or reliability files not found")
                return False
                
        except Exception as e:
            self.log_fail("Could not test message tracking", str(e))
            return False
    
    def analyze_code_structure(self):
        """Analyze the overall code structure for UDP reliability issues."""
        logger.info("🔍 Analyzing UDP Reliability Code Structure...")
        
        # Test 1: Retransmission implementation
        self.test_reliability_layer_retransmission()
        
        # Test 2: ACK forwarding
        self.test_ack_forwarding_integration()
        
        # Test 3: Message tracking consistency
        self.test_pending_message_tracking()
    
    def create_fix_recommendations(self):
        """Create specific fix recommendations."""
        logger.info("🔧 Creating Fix Recommendations...")
        
        recommendations = []
        
        if any("retransmission" in issue.lower() for issue in self.issues_found):
            recommendations.append({
                "issue": "Missing Retransmission Implementation",
                "location": "llmflow/transport/udp/reliability.py:_handle_timeout()",
                "fix": "Implement actual message retransmission by storing the send callback",
                "code": """
async def _handle_timeout(self, sequence_number: int) -> None:
    if sequence_number not in self.pending_messages:
        return
    
    pending_msg = self.pending_messages[sequence_number]
    
    if pending_msg.retries < self.max_retries:
        pending_msg.retries += 1
        pending_msg.last_retry = time.time()
        
        # FIXED: Actual retransmission implementation
        if pending_msg.send_callback:
            try:
                await pending_msg.send_callback(pending_msg.data)
                logger.debug(f"Retransmitted message {sequence_number}, attempt {pending_msg.retries}")
            except Exception as e:
                logger.error(f"Retransmission failed for {sequence_number}: {e}")
        
        self.stats['messages_retransmitted'] += 1
    else:
        del self.pending_messages[sequence_number]
        self.stats['timeouts'] += 1
        logger.warning(f"Message {sequence_number} failed after {self.max_retries} retries")
"""
            })
        
        if any("ack" in issue.lower() and "forwarding" in issue.lower() for issue in self.issues_found):
            recommendations.append({
                "issue": "ACK Not Forwarded to Reliability Layer",
                "location": "llmflow/transport/udp/transport.py:_handle_ack_message()",
                "fix": "Forward ACK to reliability layer for proper tracking",
                "code": """
async def _handle_ack_message(self, message: UDPMessage) -> None:
    # Handle acknowledgment message
    seq_num = message.sequence_number
    
    # FIXED: Forward ACK to reliability layer
    if self.reliability_layer:
        await self.reliability_layer.handle_ack(seq_num)
    
    # Keep local tracking for backward compatibility if needed
    if seq_num in self.pending_acks:
        del self.pending_acks[seq_num]
    
    logger.debug(f"Received and processed ACK for sequence {seq_num}")
"""
            })
        
        if any("dual" in issue.lower() and "tracking" in issue.lower() for issue in self.issues_found):
            recommendations.append({
                "issue": "Dual Message Tracking Systems",
                "location": "Transport and Reliability Layers",
                "fix": "Unify message tracking in reliability layer only",
                "code": """
# Remove pending_acks from transport layer and rely on reliability layer only
# Transport layer should forward all ACKs to reliability layer
# Reliability layer should be the single source of truth for message state
"""
            })
        
        return recommendations
    
    def generate_test_scenarios(self):
        """Generate specific test scenarios to verify fixes."""
        scenarios = [
            {
                "name": "Retransmission Test",
                "description": "Test that messages are retransmitted on timeout",
                "test_code": """
async def test_retransmission():
    reliability_mgr = ReliabilityManager(max_retries=3, ack_timeout=0.5)
    
    # Track sent messages
    sent_messages = []
    async def mock_send(data):
        sent_messages.append(data)
    
    # Send a message
    await reliability_mgr.send_reliable(b"test_data", mock_send)
    
    # Don't send ACK, wait for retransmission
    await asyncio.sleep(1.0)  # Wait for timeout
    
    # Should have original + retransmitted messages
    assert len(sent_messages) > 1, "Message should be retransmitted"
"""
            },
            {
                "name": "ACK Integration Test", 
                "description": "Test that ACKs from transport reach reliability layer",
                "test_code": """
async def test_ack_integration():
    transport = UDPTransport()
    reliability_mgr = transport.reliability_layer
    
    # Send message
    await reliability_mgr.send_reliable(b"test", mock_send)
    seq_num = list(reliability_mgr.pending_messages.keys())[0]
    
    # Simulate ACK received by transport
    ack_msg = UDPMessage(UDPMessageType.ACK, seq_num, 0, 1, b"")
    await transport._handle_ack_message(ack_msg)
    
    # Should be removed from pending
    assert seq_num not in reliability_mgr.pending_messages
"""
            },
            {
                "name": "Message Tracking Consistency Test",
                "description": "Test that message tracking is consistent",
                "test_code": """
async def test_tracking_consistency():
    transport = UDPTransport()
    
    # Send multiple messages
    for i in range(5):
        await transport.send(f"message_{i}".encode(), ("localhost", 8080))
    
    # Check that tracking is only in reliability layer
    reliability_pending = len(transport.reliability_layer.pending_messages)
    transport_pending = len(getattr(transport, 'pending_acks', {}))
    
    # Either transport should not track, or both should be consistent
    assert transport_pending == 0 or transport_pending == reliability_pending
"""
            }
        ]
        
        return scenarios
    
    def print_report(self):
        """Print comprehensive report."""
        print("\n" + "="*80)
        print("🔍 UDP RELIABILITY ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📊 TESTS RUN: {self.tests_run}")
        print(f"✅ TESTS PASSED: {self.tests_passed}")
        print(f"❌ ISSUES FOUND: {len(self.issues_found)}")
        
        if self.issues_found:
            print("\n🐛 CRITICAL ISSUES IDENTIFIED:")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"  {i}. {issue}")
            
            print("\n🔧 FIX RECOMMENDATIONS:")
            recommendations = self.create_fix_recommendations()
            for rec in recommendations:
                print(f"\n🎯 Issue: {rec['issue']}")
                print(f"📍 Location: {rec['location']}")
                print(f"🛠️  Fix: {rec['fix']}")
                print("💻 Code:")
                print(rec['code'])
            
            print("\n🧪 SUGGESTED TEST SCENARIOS:")
            scenarios = self.generate_test_scenarios()
            for scenario in scenarios:
                print(f"\n📋 {scenario['name']}")
                print(f"📝 {scenario['description']}")
                print("💻 Test Code:")
                print(scenario['test_code'])
        
        else:
            print("\n✅ NO CRITICAL ISSUES FOUND!")
            print("The UDP reliability implementation appears to be working correctly.")
        
        print("\n" + "="*80)
        
        return len(self.issues_found) == 0

def main():
    """Main testing function."""
    tester = UDPReliabilityTester()
    
    print("🧪 LLMFlow UDP Reliability Issues Analysis")
    print("="*50)
    
    # Analyze the code structure
    tester.analyze_code_structure()
    
    # Generate comprehensive report
    success = tester.print_report()
    
    if not success:
        print("\n⚠️  CRITICAL ISSUES FOUND!")
        print("The UDP reliability system has implementation problems that need fixing.")
        print("See the recommendations above for specific fixes.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
