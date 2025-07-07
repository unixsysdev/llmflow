"""
LLMFlow Core Framework Components

This module contains the foundational classes and interfaces for the LLMFlow framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime
from enum import Enum
import uuid


class ComponentType(Enum):
    """Types of components in the LLMFlow system."""
    DATA_ATOM = "data_atom"
    SERVICE_ATOM = "service_atom"
    MOLECULE = "molecule"
    CELL = "cell"
    ORGANISM = "organism"
    CONDUCTOR = "conductor"


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    @classmethod
    def success(cls, warnings: List[str] = None) -> 'ValidationResult':
        """Create a successful validation result."""
        return cls(True, warnings=warnings)
    
    @classmethod
    def error(cls, message: str) -> 'ValidationResult':
        """Create a failed validation result."""
        return cls(False, errors=[message])


class DataAtom(ABC):
    """Base class for all data atoms in the LLMFlow system."""
    
    def __init__(self, value: Any, metadata: Dict[str, Any] = None):
        self.value = value
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.atom_id = str(uuid.uuid4())
    
    @abstractmethod
    def validate(self) -> ValidationResult:
        """Validate the data atom's value."""
        pass
    
    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize the data atom to bytes."""
        pass
    
    @classmethod
    @abstractmethod
    def deserialize(cls, data: bytes) -> 'DataAtom':
        """Deserialize bytes to a data atom."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata for this data atom."""
        return self.metadata.copy()
    
    def get_type_signature(self) -> str:
        """Get the type signature for this data atom."""
        return f"{self.__class__.__module__}.{self.__class__.__name__}"
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.value!r}, metadata={self.metadata!r})"


class ServiceAtom(ABC):
    """Base class for all service atoms in the LLMFlow system."""
    
    def __init__(self, name: str, input_types: List[str], output_types: List[str]):
        self.name = name
        self.input_types = input_types
        self.output_types = output_types
        self.atom_id = str(uuid.uuid4())
        self.created_at = datetime.utcnow()
        self.execution_count = 0
    
    @abstractmethod
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Process input data atoms and return output data atoms."""
        pass
    
    def get_signature(self) -> 'ServiceSignature':
        """Get the service signature for this atom."""
        return ServiceSignature(
            name=self.name,
            input_types=self.input_types,
            output_types=self.output_types,
            atom_id=self.atom_id
        )
    
    def validate_inputs(self, inputs: List[DataAtom]) -> ValidationResult:
        """Validate that inputs match expected types."""
        if len(inputs) != len(self.input_types):
            return ValidationResult.error(
                f"Expected {len(self.input_types)} inputs, got {len(inputs)}"
            )
        
        for i, (input_atom, expected_type) in enumerate(zip(inputs, self.input_types)):
            if input_atom.get_type_signature() != expected_type:
                return ValidationResult.error(
                    f"Input {i}: expected {expected_type}, got {input_atom.get_type_signature()}"
                )
        
        return ValidationResult.success()
    
    def __call__(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Execute the service atom."""
        # Validate inputs
        validation_result = self.validate_inputs(inputs)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid inputs: {validation_result.errors}")
        
        # Process inputs
        self.execution_count += 1
        return self.process(inputs)


class ServiceSignature:
    """Signature definition for a service atom."""
    
    def __init__(self, name: str, input_types: List[str], output_types: List[str], 
                 atom_id: str = None, constraints: Dict[str, Any] = None, 
                 metadata: Dict[str, Any] = None):
        self.name = name
        self.input_types = input_types
        self.output_types = output_types
        self.atom_id = atom_id
        self.constraints = constraints or {}
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        inputs = ', '.join(self.input_types)
        outputs = ', '.join(self.output_types)
        return f"{self.name}({inputs}) -> ({outputs})"


class ExecutionGraph:
    """Represents the execution graph for a molecule."""
    
    def __init__(self):
        self.nodes: Dict[str, ServiceAtom] = {}
        self.edges: List[tuple] = []
        self.execution_order: List[str] = []
    
    def add_node(self, atom: ServiceAtom):
        """Add a service atom to the graph."""
        self.nodes[atom.atom_id] = atom
    
    def add_edge(self, from_atom: Type[ServiceAtom], to_atom: Type[ServiceAtom]):
        """Add an edge between two atoms."""
        self.edges.append((from_atom.__name__, to_atom.__name__))
    
    def add_sequence(self, atoms: List[Type[ServiceAtom]]):
        """Add a sequence of atoms to be executed in order."""
        for i in range(len(atoms) - 1):
            self.add_edge(atoms[i], atoms[i + 1])
    
    def add_parallel(self, atoms: List[Type[ServiceAtom]]):
        """Add atoms that can be executed in parallel."""
        # For now, just add them as nodes
        # Parallel execution logic would be implemented in the conductor
        pass
    
    def get_execution_order(self) -> List[str]:
        """Get the execution order of atoms based on dependencies."""
        # Simple topological sort would go here
        # For now, return the order they were added
        return list(self.nodes.keys())


class Component(ABC):
    """Base class for all LLMFlow components."""
    
    def __init__(self, name: str, component_type: ComponentType):
        self.name = name
        self.component_type = component_type
        self.component_id = str(uuid.uuid4())
        self.created_at = datetime.utcnow()
        self.status = "initialized"
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the component."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the component."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the component is healthy."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            "name": self.name,
            "type": self.component_type.value,
            "id": self.component_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status
        }
