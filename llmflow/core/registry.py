"""
Component Registry System

This module provides registration and discovery services for LLMFlow components.
"""

from typing import Dict, List, Type, Optional, Any
import threading
from dataclasses import dataclass
from datetime import datetime

from .base import DataAtom, ServiceAtom, Component, ComponentType, ServiceSignature


@dataclass
class ComponentRegistration:
    """Registration information for a component."""
    name: str
    component_type: ComponentType
    component_class: Type[Component]
    signature: Optional[ServiceSignature] = None
    metadata: Dict[str, Any] = None
    registered_at: datetime = None
    
    def __post_init__(self):
        if self.registered_at is None:
            self.registered_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class AtomRegistry:
    """Registry for data and service atoms."""
    
    def __init__(self):
        self._data_atoms: Dict[str, Type[DataAtom]] = {}
        self._service_atoms: Dict[str, Type[ServiceAtom]] = {}
        self._lock = threading.RLock()
    
    def register_data_atom(self, atom_class: Type[DataAtom], name: str = None) -> None:
        """Register a data atom type."""
        with self._lock:
            atom_name = name or atom_class.__name__
            if atom_name in self._data_atoms:
                raise ValueError(f"Data atom '{atom_name}' already registered")
            self._data_atoms[atom_name] = atom_class
    
    def register_service_atom(self, atom_class: Type[ServiceAtom], name: str = None) -> None:
        """Register a service atom type."""
        with self._lock:
            atom_name = name or atom_class.__name__
            if atom_name in self._service_atoms:
                raise ValueError(f"Service atom '{atom_name}' already registered")
            self._service_atoms[atom_name] = atom_class
    
    def get_data_atom(self, name: str) -> Optional[Type[DataAtom]]:
        """Get a registered data atom type."""
        with self._lock:
            return self._data_atoms.get(name)
    
    def get_service_atom(self, name: str) -> Optional[Type[ServiceAtom]]:
        """Get a registered service atom type."""
        with self._lock:
            return self._service_atoms.get(name)
    
    def list_data_atoms(self) -> List[str]:
        """List all registered data atom names."""
        with self._lock:
            return list(self._data_atoms.keys())
    
    def list_service_atoms(self) -> List[str]:
        """List all registered service atom names."""
        with self._lock:
            return list(self._service_atoms.keys())
    
    def validate_atom_compatibility(self, source_type: str, target_type: str) -> bool:
        """Validate if two atom types are compatible."""
        with self._lock:
            # Simple type equality check for now
            # In a full implementation, this would include type hierarchy and conversion rules
            return source_type == target_type
    
    def get_atom_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered atom."""
        with self._lock:
            data_atom = self._data_atoms.get(name)
            service_atom = self._service_atoms.get(name)
            
            if data_atom:
                return {
                    "name": name,
                    "type": "data_atom",
                    "class": data_atom.__name__,
                    "module": data_atom.__module__
                }
            elif service_atom:
                return {
                    "name": name,
                    "type": "service_atom",
                    "class": service_atom.__name__,
                    "module": service_atom.__module__
                }
            return None


class ServiceRegistry:
    """Registry for service signatures and instances."""
    
    def __init__(self):
        self._services: Dict[str, ServiceSignature] = {}
        self._instances: Dict[str, ServiceAtom] = {}
        self._lock = threading.RLock()
    
    def register_service(self, service: ServiceAtom) -> None:
        """Register a service atom instance."""
        with self._lock:
            signature = service.get_signature()
            if signature.name in self._services:
                raise ValueError(f"Service '{signature.name}' already registered")
            
            self._services[signature.name] = signature
            self._instances[signature.atom_id] = service
    
    def get_service_signature(self, name: str) -> Optional[ServiceSignature]:
        """Get a service signature by name."""
        with self._lock:
            return self._services.get(name)
    
    def get_service_instance(self, atom_id: str) -> Optional[ServiceAtom]:
        """Get a service instance by atom ID."""
        with self._lock:
            return self._instances.get(atom_id)
    
    def list_services(self) -> List[str]:
        """List all registered service names."""
        with self._lock:
            return list(self._services.keys())
    
    def find_compatible_services(self, input_types: List[str], output_types: List[str]) -> List[ServiceSignature]:
        """Find services compatible with given input/output types."""
        with self._lock:
            compatible = []
            for signature in self._services.values():
                if (signature.input_types == input_types and 
                    signature.output_types == output_types):
                    compatible.append(signature)
            return compatible


class ComponentRegistry:
    """Master registry for all LLMFlow components."""
    
    def __init__(self):
        self._components: Dict[str, ComponentRegistration] = {}
        self._instances: Dict[str, Component] = {}
        self._lock = threading.RLock()
        self.atom_registry = AtomRegistry()
        self.service_registry = ServiceRegistry()
    
    def register_component(self, component_class: Type[Component], 
                         name: str = None, metadata: Dict[str, Any] = None) -> None:
        """Register a component class."""
        with self._lock:
            component_name = name or component_class.__name__
            
            if component_name in self._components:
                raise ValueError(f"Component '{component_name}' already registered")
            
            # Determine component type
            component_type = self._determine_component_type(component_class)
            
            registration = ComponentRegistration(
                name=component_name,
                component_type=component_type,
                component_class=component_class,
                metadata=metadata or {}
            )
            
            self._components[component_name] = registration
    
    def register_component_instance(self, component: Component) -> None:
        """Register a component instance."""
        with self._lock:
            if component.component_id in self._instances:
                raise ValueError(f"Component instance '{component.component_id}' already registered")
            
            self._instances[component.component_id] = component
    
    def get_component_class(self, name: str) -> Optional[Type[Component]]:
        """Get a component class by name."""
        with self._lock:
            registration = self._components.get(name)
            return registration.component_class if registration else None
    
    def get_component_instance(self, component_id: str) -> Optional[Component]:
        """Get a component instance by ID."""
        with self._lock:
            return self._instances.get(component_id)
    
    def list_components(self, component_type: ComponentType = None) -> List[str]:
        """List all registered component names, optionally filtered by type."""
        with self._lock:
            if component_type is None:
                return list(self._components.keys())
            else:
                return [name for name, reg in self._components.items() 
                       if reg.component_type == component_type]
    
    def list_instances(self, component_type: ComponentType = None) -> List[str]:
        """List all registered component instances, optionally filtered by type."""
        with self._lock:
            if component_type is None:
                return list(self._instances.keys())
            else:
                return [comp_id for comp_id, comp in self._instances.items()
                       if comp.component_type == component_type]
    
    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered component."""
        with self._lock:
            registration = self._components.get(name)
            if not registration:
                return None
            
            return {
                "name": registration.name,
                "type": registration.component_type.value,
                "class": registration.component_class.__name__,
                "module": registration.component_class.__module__,
                "registered_at": registration.registered_at.isoformat(),
                "metadata": registration.metadata
            }
    
    def _determine_component_type(self, component_class: Type[Component]) -> ComponentType:
        """Determine the type of a component class."""
        class_name = component_class.__name__.lower()
        
        if "atom" in class_name:
            if issubclass(component_class, ServiceAtom):
                return ComponentType.SERVICE_ATOM
            elif issubclass(component_class, DataAtom):
                return ComponentType.DATA_ATOM
        elif "molecule" in class_name:
            return ComponentType.MOLECULE
        elif "cell" in class_name:
            return ComponentType.CELL
        elif "organism" in class_name:
            return ComponentType.ORGANISM
        elif "conductor" in class_name:
            return ComponentType.CONDUCTOR
        
        # Default to service atom for unknown types
        return ComponentType.SERVICE_ATOM


# Global registry instance
_global_registry = ComponentRegistry()


def get_global_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _global_registry


def register_component(component_class: Type[Component], name: str = None, 
                      metadata: Dict[str, Any] = None) -> None:
    """Register a component with the global registry."""
    _global_registry.register_component(component_class, name, metadata)


def register_data_atom(atom_class: Type[DataAtom], name: str = None) -> None:
    """Register a data atom with the global registry."""
    _global_registry.atom_registry.register_data_atom(atom_class, name)


def register_service_atom(atom_class: Type[ServiceAtom], name: str = None) -> None:
    """Register a service atom with the global registry."""
    _global_registry.atom_registry.register_service_atom(atom_class, name)
