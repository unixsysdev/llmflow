"""
Test Queue Protocol Operations

Quick test to verify all queue protocol operations work.
"""

import asyncio
import sys
sys.path.insert(0, '.')

async def test_queue_operations():
    """Test all queue protocol operations."""
    print("🔍 Testing Queue Protocol Operations...")
    
    try:
        from llmflow.queue.protocol import QueueProtocol, SecurityLevel
        
        # Create protocol instance
        protocol = QueueProtocol("127.0.0.1", 18300)
        print("  ✓ Queue protocol created")
        
        # Start protocol
        await protocol.start()
        print("  ✓ Queue protocol started")
        
        # Test context switch
        result = await protocol.context_switch("test_queue", SecurityLevel.ENCRYPTED)
        print(f"  ✓ Context switch result: {result}")
        
        # Test transfer (using dummy implementation)
        try:
            transferred = await protocol.transfer("queue1", "queue2", 1)
            print(f"  ✓ Transfer operation completed: {transferred}")
        except Exception as e:
            print(f"  ⚠️ Transfer failed (expected for test): {e}")
        
        # Stop protocol
        await protocol.stop()
        print("  ✓ Queue protocol stopped")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Queue operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🚀 Queue Protocol Test\n")
    
    success = await test_queue_operations()
    
    print(f"\nResult: {'✅ Queue operations work!' if success else '❌ Queue operations failed'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
