"""
Basic tests for LLMFlow framework components (without pytest).
"""

import asyncio
from datetime import datetime

from llmflow.core.base import ValidationResult
from llmflow.atoms.data import StringAtom, IntegerAtom, BooleanAtom, EmailAtom
from llmflow.atoms.service import ValidateEmailAtom
from llmflow.queue import QueueManager
from llmflow.molecules.optimization import PerformanceMetrics, PerformanceMetricsAtom


def test_data_atoms():
    """Test data atom functionality."""
    print("Testing data atoms...")
    
    # Test string atom
    atom = StringAtom("test string")
    assert atom.value == "test string"
    result = atom.validate()
    assert result.is_valid
    
    # Test integer atom
    atom = IntegerAtom(42)
    assert atom.value == 42
    result = atom.validate()
    assert result.is_valid
    
    # Test boolean atom
    atom = BooleanAtom(True)
    assert atom.value is True
    result = atom.validate()
    assert result.is_valid
    
    # Test serialization
    atom = StringAtom("test")
    serialized = atom.serialize()
    assert isinstance(serialized, bytes)
    
    deserialized = StringAtom.deserialize(serialized)
    assert deserialized.value == atom.value
    
    print("✓ Data atoms tests passed")


def test_service_atoms():
    """Test service atom functionality."""
    print("Testing service atoms...")
    
    validator = ValidateEmailAtom()
    
    # Test valid email
    valid_email = EmailAtom("test@example.com")
    result = validator.process([valid_email])
    
    assert len(result) == 1
    assert isinstance(result[0], BooleanAtom)
    assert result[0].value is True
    
    # Test invalid email  
    invalid_email = EmailAtom("invalid-email")
    result = validator.process([invalid_email])
    
    assert len(result) == 1
    assert isinstance(result[0], BooleanAtom)
    assert result[0].value is False
    
    print("✓ Service atoms tests passed")


async def test_queue_system():
    """Test queue system functionality."""
    print("Testing queue system...")
    
    manager = QueueManager()
    await manager.start()
    
    try:
        # Create queue
        success = await manager.create_queue("test_queue")
        assert success is True
        
        # Test queue listing
        queues = await manager.list_queues()
        assert "test_queue" in queues
        
        # Test enqueue
        test_data = {"message": "hello world"}
        message_id = await manager.enqueue("test_queue", test_data)
        assert message_id is not None
        
        # Test dequeue
        dequeued_data = await manager.dequeue("test_queue")
        assert dequeued_data == test_data
        
    finally:
        await manager.stop()
    
    print("✓ Queue system tests passed")


def test_optimization_molecules():
    """Test optimization molecule functionality."""
    print("Testing optimization molecules...")
    
    metrics = PerformanceMetrics(
        timestamp=datetime.utcnow(),
        latency_ms=100.0,
        throughput_ops_per_sec=1000.0,
        error_rate=0.01,
        memory_usage_mb=256.0,
        cpu_usage_percent=50.0,
        queue_depth=10,
        processed_messages=5000
    )
    
    atom = PerformanceMetricsAtom(metrics)
    assert atom.metrics == metrics
    
    # Test validation
    result = atom.validate()
    assert result.is_valid
    
    # Test serialization
    serialized = atom.serialize()
    assert isinstance(serialized, bytes)
    
    # Test deserialization
    deserialized = PerformanceMetricsAtom.deserialize(serialized)
    assert deserialized.metrics.latency_ms == metrics.latency_ms
    
    print("✓ Optimization molecules tests passed")


def test_validation_results():
    """Test validation result functionality."""
    print("Testing validation results...")
    
    # Test success result
    result = ValidationResult.success()
    assert result.is_valid
    assert len(result.errors) == 0
    
    # Test error result
    result = ValidationResult.error("Test error")
    assert not result.is_valid
    assert "Test error" in result.errors
    
    # Test result with warnings
    result = ValidationResult.success(warnings=["Test warning"])
    assert result.is_valid
    assert "Test warning" in result.warnings
    
    print("✓ Validation results tests passed")


async def main():
    """Run all tests."""
    print("🚀 Starting LLMFlow basic tests...\n")
    
    try:
        # Run synchronous tests
        test_data_atoms()
        test_service_atoms()
        test_validation_results()
        test_optimization_molecules()
        
        # Run asynchronous tests
        await test_queue_system()
        
        print("\n🎉 All basic tests passed!")
        print("✅ LLMFlow framework is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
