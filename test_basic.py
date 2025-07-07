"""
Basic tests for LLMFlow framework components.
"""

import asyncio
import pytest
from datetime import datetime

from llmflow.core.base import ValidationResult
from llmflow.atoms.data import StringAtom, IntegerAtom, BooleanAtom
from llmflow.atoms.service import ValidateEmailAtom
from llmflow.queue import QueueManager, QueueClient
from llmflow.molecules.optimization import PerformanceMetrics, PerformanceMetricsAtom


class TestDataAtoms:
    """Test data atom functionality."""
    
    def test_string_atom_creation(self):
        """Test string atom creation and validation."""
        atom = StringAtom("test string")
        assert atom.value == "test string"
        
        result = atom.validate()
        assert result.is_valid
    
    def test_integer_atom_creation(self):
        """Test integer atom creation and validation."""
        atom = IntegerAtom(42)
        assert atom.value == 42
        
        result = atom.validate()
        assert result.is_valid
    
    def test_boolean_atom_creation(self):
        """Test boolean atom creation and validation."""
        atom = BooleanAtom(True)
        assert atom.value is True
        
        result = atom.validate()
        assert result.is_valid
    
    def test_atom_serialization(self):
        """Test atom serialization and deserialization."""
        atom = StringAtom("test")
        serialized = atom.serialize()
        
        # Verify serialization produces bytes
        assert isinstance(serialized, bytes)
        
        # Test deserialization
        deserialized = StringAtom.deserialize(serialized)
        assert deserialized.value == atom.value


class TestServiceAtoms:
    """Test service atom functionality."""
    
    def test_email_validation(self):
        """Test email validation service atom."""
        validator = ValidateEmailAtom()
        
        # Test valid email
        valid_email = StringAtom("test@example.com")
        result = validator.process([valid_email])
        
        assert len(result) == 1
        assert isinstance(result[0], BooleanAtom)
        assert result[0].value is True
        
        # Test invalid email
        invalid_email = StringAtom("invalid-email")
        result = validator.process([invalid_email])
        
        assert len(result) == 1
        assert isinstance(result[0], BooleanAtom)
        assert result[0].value is False


class TestQueueSystem:
    """Test queue system functionality."""
    
    @pytest.mark.asyncio
    async def test_queue_manager_creation(self):
        """Test queue manager creation."""
        manager = QueueManager()
        assert manager is not None
        
        # Test queue creation
        success = await manager.create_queue("test_queue")
        assert success is True
        
        # Test queue listing
        queues = await manager.list_queues()
        assert "test_queue" in queues
    
    @pytest.mark.asyncio
    async def test_queue_operations(self):
        """Test basic queue operations."""
        manager = QueueManager()
        await manager.start()
        
        try:
            # Create queue
            await manager.create_queue("test_queue")
            
            # Test enqueue
            test_data = {"message": "hello world"}
            message_id = await manager.enqueue("test_queue", test_data)
            assert message_id is not None
            
            # Test dequeue
            dequeued_data = await manager.dequeue("test_queue")
            assert dequeued_data == test_data
            
        finally:
            await manager.stop()


class TestOptimizationMolecules:
    """Test optimization molecule functionality."""
    
    def test_performance_metrics_atom(self):
        """Test performance metrics atom creation."""
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


class TestValidationResult:
    """Test validation result functionality."""
    
    def test_success_result(self):
        """Test successful validation result."""
        result = ValidationResult.success()
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_error_result(self):
        """Test error validation result."""
        result = ValidationResult.error("Test error")
        assert not result.is_valid
        assert "Test error" in result.errors
    
    def test_result_with_warnings(self):
        """Test validation result with warnings."""
        result = ValidationResult.success(warnings=["Test warning"])
        assert result.is_valid
        assert "Test warning" in result.warnings


if __name__ == "__main__":
    # Run basic tests
    import sys
    
    # Test data atoms
    test_data_atoms = TestDataAtoms()
    test_data_atoms.test_string_atom_creation()
    test_data_atoms.test_integer_atom_creation()
    test_data_atoms.test_boolean_atom_creation()
    test_data_atoms.test_atom_serialization()
    print("✓ Data atoms tests passed")
    
    # Test service atoms
    test_service_atoms = TestServiceAtoms()
    test_service_atoms.test_email_validation()
    print("✓ Service atoms tests passed")
    
    # Test validation results
    test_validation = TestValidationResult()
    test_validation.test_success_result()
    test_validation.test_error_result()
    test_validation.test_result_with_warnings()
    print("✓ Validation result tests passed")
    
    # Test optimization molecules
    test_optimization = TestOptimizationMolecules()
    test_optimization.test_performance_metrics_atom()
    print("✓ Optimization molecule tests passed")
    
    print("\n🎉 All basic tests passed!")
    print("Note: Async tests require pytest-asyncio to run properly")
