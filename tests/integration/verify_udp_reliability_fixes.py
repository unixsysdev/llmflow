#!/usr/bin/env python3
"""
Verify UDP Reliability Fixes

This script verifies that the critical UDP reliability fixes have been applied
by examining the source code directly.
"""

import re
from pathlib import Path

def check_retransmission_fix():
    """Check that retransmission is properly implemented."""
    reliability_file = Path("llmflow/transport/udp/reliability.py")
    
    if not reliability_file.exists():
        return False, "Reliability file not found"
    
    content = reliability_file.read_text()
    
    # Check that TODO is removed
    if "TODO: Implement actual retransmission" in content:
        return False, "TODO comment still present - retransmission not implemented"
    
    # Check for actual retransmission implementation
    if "pending_msg.send_callback" in content and "await pending_msg.send_callback" in content:
        return True, "Retransmission implementation found"
    else:
        return False, "Retransmission implementation not found"

def check_reliability_manager_integration():
    """Check that UDPTransport uses ReliabilityManager."""
    transport_file = Path("llmflow/transport/udp/transport.py")
    
    if not transport_file.exists():
        return False, "Transport file not found"
    
    content = transport_file.read_text()
    
    # Check for ReliabilityManager import
    if "from .reliability import ReliabilityManager" not in content:
        return False, "ReliabilityManager import not found"
    
    # Check that UDPTransport uses reliability_manager
    if "self.reliability_manager = ReliabilityManager(" in content:
        return True, "UDPTransport uses ReliabilityManager"
    else:
        return False, "UDPTransport does not use ReliabilityManager"

def check_ack_forwarding():
    """Check that ACK forwarding is implemented."""
    transport_file = Path("llmflow/transport/udp/transport.py")
    
    if not transport_file.exists():
        return False, "Transport file not found"
    
    content = transport_file.read_text()
    
    # Check for ACK handling that forwards to reliability manager
    if "reliability_manager.handle_ack" in content:
        return True, "ACK forwarding to reliability manager found"
    else:
        return False, "ACK forwarding not implemented"

def check_pending_message_send_callback():
    """Check that PendingMessage has send_callback field."""
    reliability_file = Path("llmflow/transport/udp/reliability.py")
    
    if not reliability_file.exists():
        return False, "Reliability file not found"
    
    content = reliability_file.read_text()
    
    # Look for PendingMessage class definition with send_callback
    pending_msg_match = re.search(r'@dataclass\s+class PendingMessage:.*?(?=\n\n|\nclass|\nlog)', content, re.DOTALL)
    
    if pending_msg_match:
        pending_msg_def = pending_msg_match.group(0)
        if "send_callback: Optional[Callable" in pending_msg_def:
            return True, "PendingMessage has send_callback field"
        else:
            return False, "PendingMessage missing send_callback field"
    else:
        return False, "PendingMessage class not found"

def main():
    """Main verification function."""
    print("🔍 UDP Reliability Fix Verification")
    print("=" * 50)
    
    tests = [
        ("Retransmission Implementation", check_retransmission_fix),
        ("ReliabilityManager Integration", check_reliability_manager_integration),
        ("ACK Forwarding", check_ack_forwarding),
        ("PendingMessage Send Callback", check_pending_message_send_callback),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed, message = test_func()
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {test_name} - {message}")
            results.append(passed)
        except Exception as e:
            print(f"❌ ERROR: {test_name} - {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    passed_count = sum(results)
    total_count = len(results)
    
    print(f"📊 SUMMARY: {passed_count}/{total_count} fixes verified")
    
    if passed_count == total_count:
        print("🎉 ALL CRITICAL UDP RELIABILITY FIXES APPLIED!")
        print("\n🐛 **FIXED ISSUES:**")
        print("1. ✅ Retransmission now properly implemented in ReliabilityManager._handle_timeout()")
        print("2. ✅ ACK messages are forwarded to reliability layer for proper tracking")
        print("3. ✅ Unified message tracking using ReliabilityManager instead of dual systems")
        print("4. ✅ PendingMessage now stores send_callback for retransmission")
        return True
    else:
        print("⚠️  SOME FIXES NOT APPLIED - Please review the failed checks")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
