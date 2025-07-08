"""
LLMFlow Graph Definition System

This module provides the core graph definition system that allows users to define
applications as connected atoms, molecules, and cells. The LLM then generates
working components from these definitions.
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from ..core.base import ComponentType, ValidationResult


class ConnectionType(Enum):
    """Types of connections between components."""
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow" 
    EVENT_TRIGGER = "event_trigger"
    FEEDBACK_LOOP = "feedback_loop"


@dataclass
class ComponentSpec:
    """Specification for a single component in the graph."""
    
    # Basic info
    id: str
    name: str
    component_type: ComponentType
    description: str
    
    # Input/Output definitions
    input_types: List[str]
    output_types: List[str]
    
    # Queue configuration
    input_queues: List[str]
    output_queues: List[str]
    
    # Performance requirements
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    
    # Implementation hints for LLM
    implementation_hints: Dict[str, Any] = None
    
    # Dependencies
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.implementation_hints is None:
            self.implementation_hints = {}
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ConnectionSpec:
    """Specification for a connection between components."""
    
    id: str
    source_component: str
    target_component: str
    source_queue: str
    target_queue: str
    connection_type: ConnectionType
    data_types: List[str]
    
    # Connection properties
    buffer_size: int = 1000
    timeout_ms: int = 5000
    retry_attempts: int = 3
    
    # Validation rules
    validation_rules: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = {}


@dataclass
class GraphDefinition:
    """Complete definition of an LLMFlow application graph."""
    
    # Graph metadata
    id: str
    name: str
    description: str
    version: str
    created_at: str
    
    # Components and connections
    components: Dict[str, ComponentSpec]
    connections: Dict[str, ConnectionSpec]
    
    # Global configuration
    global_config: Dict[str, Any]
    
    # Performance goals
    performance_goals: Dict[str, Any]
    
    def __post_init__(self):
        if not self.components:
            self.components = {}
        if not self.connections:
            self.connections = {}
        if not self.global_config:
            self.global_config = {}
        if not self.performance_goals:
            self.performance_goals = {}
    
    def add_component(self, component: ComponentSpec) -> None:
        """Add a component to the graph."""
        self.components[component.id] = component
    
    def add_connection(self, connection: ConnectionSpec) -> None:
        """Add a connection to the graph."""
        self.connections[connection.id] = connection
    
    def get_component(self, component_id: str) -> Optional[ComponentSpec]:
        """Get a component by ID."""
        return self.components.get(component_id)
    
    def get_connection(self, connection_id: str) -> Optional[ConnectionSpec]:
        """Get a connection by ID."""
        return self.connections.get(connection_id)
    
    def get_components_by_type(self, component_type: ComponentType) -> List[ComponentSpec]:
        """Get all components of a specific type."""
        return [comp for comp in self.components.values() if comp.component_type == component_type]
    
    def get_component_dependencies(self, component_id: str) -> List[str]:
        """Get all components that this component depends on."""
        dependencies = []
        
        # Direct dependencies
        component = self.get_component(component_id)
        if component:
            dependencies.extend(component.dependencies)
        
        # Connection-based dependencies
        for connection in self.connections.values():
            if connection.target_component == component_id:
                dependencies.append(connection.source_component)
        
        return list(set(dependencies))
    
    def get_deployment_order(self) -> List[str]:
        """Get the order in which components should be deployed."""
        deployment_order = []
        remaining_components = set(self.components.keys())
        
        while remaining_components:
            # Find components with no undeployed dependencies
            ready_components = []
            
            for component_id in remaining_components:
                dependencies = self.get_component_dependencies(component_id)
                if all(dep in deployment_order or dep not in self.components for dep in dependencies):
                    ready_components.append(component_id)
            
            if not ready_components:
                # Circular dependency - break it by deploying atoms first
                atoms = [cid for cid in remaining_components 
                        if self.components[cid].component_type == ComponentType.ATOM]
                if atoms:
                    ready_components = atoms[:1]
                else:
                    # Just pick one to break the cycle
                    ready_components = [list(remaining_components)[0]]
            
            # Sort by component type (atoms first, then molecules, then cells)
            ready_components.sort(key=lambda cid: self.components[cid].component_type.value)
            
            deployment_order.extend(ready_components)
            remaining_components -= set(ready_components)
        
        return deployment_order
    
    def validate(self) -> ValidationResult:
        """Validate the graph definition."""
        errors = []
        
        # Validate components
        for component_id, component in self.components.items():
            # Check component ID consistency
            if component.id != component_id:
                errors.append(f"Component ID mismatch: {component_id} vs {component.id}")
            
            # Validate queues
            if not component.input_queues and component.input_types:
                errors.append(f"Component {component_id} has input types but no input queues")
            
            if not component.output_queues and component.output_types:
                errors.append(f"Component {component_id} has output types but no output queues")
        
        # Validate connections
        for connection_id, connection in self.connections.items():
            # Check connection ID consistency
            if connection.id != connection_id:
                errors.append(f"Connection ID mismatch: {connection_id} vs {connection.id}")
            
            # Check that referenced components exist
            if connection.source_component not in self.components:
                errors.append(f"Connection {connection_id} references unknown source component: {connection.source_component}")
            
            if connection.target_component not in self.components:
                errors.append(f"Connection {connection_id} references unknown target component: {connection.target_component}")
            
            # Check queue compatibility
            if connection.source_component in self.components:
                source_comp = self.components[connection.source_component]
                if connection.source_queue not in source_comp.output_queues:
                    errors.append(f"Connection {connection_id} uses non-existent source queue: {connection.source_queue}")
            
            if connection.target_component in self.components:
                target_comp = self.components[connection.target_component]
                if connection.target_queue not in target_comp.input_queues:
                    errors.append(f"Connection {connection_id} uses non-existent target queue: {connection.target_queue}")
        
        # Check for circular dependencies
        try:
            self.get_deployment_order()
        except Exception as e:
            errors.append(f"Circular dependency detected: {e}")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at,
            'components': {
                cid: {
                    **asdict(comp),
                    'component_type': comp.component_type.value
                } for cid, comp in self.components.items()
            },
            'connections': {
                cid: {
                    **asdict(conn),
                    'connection_type': conn.connection_type.value
                } for cid, conn in self.connections.items()
            },
            'global_config': self.global_config,
            'performance_goals': self.performance_goals
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphDefinition':
        """Create from dictionary."""
        # Convert components
        components = {}
        for cid, comp_data in data.get('components', {}).items():
            comp_data = comp_data.copy()
            comp_data['component_type'] = ComponentType(comp_data['component_type'])
            components[cid] = ComponentSpec(**comp_data)
        
        # Convert connections
        connections = {}
        for cid, conn_data in data.get('connections', {}).items():
            conn_data = conn_data.copy()
            conn_data['connection_type'] = ConnectionType(conn_data['connection_type'])
            connections[cid] = ConnectionSpec(**conn_data)
        
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            version=data['version'],
            created_at=data['created_at'],
            components=components,
            connections=connections,
            global_config=data.get('global_config', {}),
            performance_goals=data.get('performance_goals', {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'GraphDefinition':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class GraphBuilder:
    """Builder for creating graph definitions."""
    
    def __init__(self, name: str, description: str = ""):
        self.graph = GraphDefinition(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            version="1.0.0",
            created_at=datetime.utcnow().isoformat(),
            components={},
            connections={},
            global_config={},
            performance_goals={}
        )
    
    def add_atom(self, name: str, description: str = "", 
                 input_types: List[str] = None, output_types: List[str] = None,
                 input_queues: List[str] = None, output_queues: List[str] = None,
                 **kwargs) -> str:
        """Add an atom component."""
        component_id = f"atom_{len([c for c in self.graph.components.values() if c.component_type == ComponentType.ATOM]) + 1}"
        
        component = ComponentSpec(
            id=component_id,
            name=name,
            component_type=ComponentType.ATOM,
            description=description,
            input_types=input_types or [],
            output_types=output_types or [],
            input_queues=input_queues or [],
            output_queues=output_queues or [],
            **kwargs
        )
        
        self.graph.add_component(component)
        return component_id
    
    def add_molecule(self, name: str, description: str = "",
                     input_types: List[str] = None, output_types: List[str] = None,
                     input_queues: List[str] = None, output_queues: List[str] = None,
                     **kwargs) -> str:
        """Add a molecule component."""
        component_id = f"molecule_{len([c for c in self.graph.components.values() if c.component_type == ComponentType.MOLECULE]) + 1}"
        
        component = ComponentSpec(
            id=component_id,
            name=name,
            component_type=ComponentType.MOLECULE,
            description=description,
            input_types=input_types or [],
            output_types=output_types or [],
            input_queues=input_queues or [],
            output_queues=output_queues or [],
            **kwargs
        )
        
        self.graph.add_component(component)
        return component_id
    
    def add_cell(self, name: str, description: str = "",
                 input_types: List[str] = None, output_types: List[str] = None,
                 input_queues: List[str] = None, output_queues: List[str] = None,
                 **kwargs) -> str:
        """Add a cell component."""
        component_id = f"cell_{len([c for c in self.graph.components.values() if c.component_type == ComponentType.CELL]) + 1}"
        
        component = ComponentSpec(
            id=component_id,
            name=name,
            component_type=ComponentType.CELL,
            description=description,
            input_types=input_types or [],
            output_types=output_types or [],
            input_queues=input_queues or [],
            output_queues=output_queues or [],
            **kwargs
        )
        
        self.graph.add_component(component)
        return component_id
    
    def connect(self, source_component: str, target_component: str,
                source_queue: str, target_queue: str,
                data_types: List[str], connection_type: ConnectionType = ConnectionType.DATA_FLOW,
                **kwargs) -> str:
        """Add a connection between components."""
        connection_id = f"conn_{len(self.graph.connections) + 1}"
        
        connection = ConnectionSpec(
            id=connection_id,
            source_component=source_component,
            target_component=target_component,
            source_queue=source_queue,
            target_queue=target_queue,
            connection_type=connection_type,
            data_types=data_types,
            **kwargs
        )
        
        self.graph.add_connection(connection)
        return connection_id
    
    def set_performance_goals(self, **goals):
        """Set performance goals for the application."""
        self.graph.performance_goals.update(goals)
    
    def set_config(self, **config):
        """Set global configuration."""
        self.graph.global_config.update(config)
    
    def build(self) -> GraphDefinition:
        """Build and validate the graph."""
        validation = self.graph.validate()
        if not validation.is_valid:
            raise ValueError(f"Graph validation failed: {validation.errors}")
        
        return self.graph


def create_clock_app_graph() -> GraphDefinition:
    """Create a sample clock application graph."""
    builder = GraphBuilder("LLMFlow Clock App", "Real-time clock application with graph-based architecture")
    
    # Atoms (data types)
    time_atom = builder.add_atom(
        name="TimeAtom",
        description="Atomic time data type with validation",
        output_types=["time_data"],
        output_queues=["time_output"],
        implementation_hints={
            "data_type": "timestamp",
            "validation": "timezone-aware datetime",
            "serialization": "ISO format"
        }
    )
    
    clock_state_atom = builder.add_atom(
        name="ClockStateAtom", 
        description="Clock state management atom",
        input_types=["time_data"],
        output_types=["clock_state"],
        input_queues=["state_input"],
        output_queues=["state_output"],
        implementation_hints={
            "state_fields": ["current_time", "timezone", "format", "is_running"],
            "persistence": "in-memory with backup"
        }
    )
    
    # Service atoms
    time_formatter = builder.add_atom(
        name="TimeFormatterAtom",
        description="Time formatting service atom",
        input_types=["time_data"],
        output_types=["formatted_time"],
        input_queues=["format_input"],
        output_queues=["format_output"],
        implementation_hints={
            "formats": ["HH:MM:SS", "12-hour", "24-hour", "ISO"],
            "localization": "timezone-aware"
        }
    )
    
    # Molecules (business logic)
    clock_logic = builder.add_molecule(
        name="ClockLogicMolecule",
        description="Core clock business logic",
        input_types=["time_data", "clock_state"],
        output_types=["time_update", "state_update"],
        input_queues=["logic_time_input", "logic_state_input"],
        output_queues=["logic_time_output", "logic_state_output"],
        dependencies=[time_atom, clock_state_atom],
        implementation_hints={
            "functionality": "real-time updates, timezone conversion, format management",
            "update_interval": "1 second",
            "error_handling": "graceful degradation"
        }
    )
    
    display_molecule = builder.add_molecule(
        name="DisplayMolecule",
        description="Display management molecule",
        input_types=["formatted_time", "state_update"],
        output_types=["display_command"],
        input_queues=["display_time_input", "display_state_input"],
        output_queues=["display_output"],
        dependencies=[time_formatter, clock_logic],
        implementation_hints={
            "display_types": ["console", "web", "api"],
            "refresh_strategy": "delta updates only",
            "styling": "configurable themes"
        }
    )
    
    # Cell (application)
    clock_app = builder.add_cell(
        name="ClockApplicationCell",
        description="Complete clock application orchestrator",
        input_types=["display_command"],
        output_types=["app_status"],
        input_queues=["app_input"],
        output_queues=["app_status_output"],
        dependencies=[clock_logic, display_molecule],
        implementation_hints={
            "orchestration": "coordinate all components",
            "lifecycle": "startup, running, shutdown",
            "monitoring": "health checks and metrics"
        }
    )
    
    # Connections
    builder.connect(
        time_atom, clock_state_atom,
        "time_output", "state_input",
        ["time_data"],
        ConnectionType.DATA_FLOW
    )
    
    builder.connect(
        time_atom, time_formatter,
        "time_output", "format_input", 
        ["time_data"],
        ConnectionType.DATA_FLOW
    )
    
    builder.connect(
        clock_state_atom, clock_logic,
        "state_output", "logic_state_input",
        ["clock_state"],
        ConnectionType.DATA_FLOW
    )
    
    builder.connect(
        time_formatter, display_molecule,
        "format_output", "display_time_input",
        ["formatted_time"],
        ConnectionType.DATA_FLOW
    )
    
    builder.connect(
        clock_logic, display_molecule,
        "logic_state_output", "display_state_input",
        ["state_update"],
        ConnectionType.DATA_FLOW
    )
    
    builder.connect(
        display_molecule, clock_app,
        "display_output", "app_input",
        ["display_command"],
        ConnectionType.DATA_FLOW
    )
    
    # Performance goals
    builder.set_performance_goals(
        max_latency_ms=50,
        target_throughput_ops_per_sec=1000,
        max_memory_mb=100,
        max_cpu_percent=10
    )
    
    # Configuration
    builder.set_config(
        default_timezone="UTC",
        update_interval_ms=1000,
        display_format="24-hour",
        queue_buffer_size=1000,
        enable_monitoring=True
    )
    
    return builder.build()


if __name__ == "__main__":
    # Demo the graph definition system
    print("🏗️ LLMFlow Graph Definition System Demo")
    print("=" * 50)
    
    # Create clock app graph
    graph = create_clock_app_graph()
    
    print(f"📱 Created graph: {graph.name}")
    print(f"   Components: {len(graph.components)}")
    print(f"   Connections: {len(graph.connections)}")
    
    # Validate
    validation = graph.validate()
    print(f"   Validation: {'✅ Valid' if validation.is_valid else '❌ Invalid'}")
    if not validation.is_valid:
        for error in validation.errors:
            print(f"     - {error}")
    
    # Show deployment order
    deployment_order = graph.get_deployment_order()
    print(f"   Deployment order: {' → '.join(deployment_order)}")
    
    # Export to JSON
    graph_json = graph.to_json()
    print(f"   JSON size: {len(graph_json)} characters")
    
    # Show component details
    print(f"\n📋 Components:")
    for comp_id, comp in graph.components.items():
        print(f"   {comp.component_type.value}: {comp.name}")
        print(f"     Inputs: {comp.input_types} → {comp.input_queues}")
        print(f"     Outputs: {comp.output_types} → {comp.output_queues}")
    
    print(f"\n🔗 Connections:")
    for conn_id, conn in graph.connections.items():
        print(f"   {conn.source_component}.{conn.source_queue} → {conn.target_component}.{conn.target_queue}")
        print(f"     Type: {conn.connection_type.value}, Data: {conn.data_types}")
    
    print(f"\n💾 Saving to clock_app_graph.json...")
    with open("clock_app_graph.json", "w") as f:
        f.write(graph_json)
    
    print("✅ Graph definition system working!")
