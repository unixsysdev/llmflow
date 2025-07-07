"""
Component Lifecycle Management

This module provides lifecycle management for LLMFlow components.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Callable
import asyncio
import logging
from datetime import datetime

from .base import Component, ComponentType


class ComponentState(Enum):
    """Possible states of a component."""
    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    FAILED = "failed"


class LifecycleEvent:
    """Represents a lifecycle event for a component."""
    
    def __init__(self, component_id: str, event_type: str, 
                 old_state: ComponentState, new_state: ComponentState,
                 metadata: Dict[str, Any] = None):
        self.component_id = component_id
        self.event_type = event_type
        self.old_state = old_state
        self.new_state = new_state
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()


class ComponentLifecycleManager:
    """Manages the lifecycle of components."""
    
    def __init__(self):
        self._components: Dict[str, Component] = {}
        self._states: Dict[str, ComponentState] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._logger = logging.getLogger(__name__)
    
    def register_component(self, component: Component, config: Dict[str, Any] = None) -> None:
        """Register a component for lifecycle management."""
        component_id = component.component_id
        
        if component_id in self._components:
            raise ValueError(f"Component {component_id} already registered")
        
        self._components[component_id] = component
        self._states[component_id] = ComponentState.CREATED
        self._configs[component_id] = config or {}
        
        self._logger.info(f"Registered component {component_id} ({component.name})")
    
    def get_component_state(self, component_id: str) -> Optional[ComponentState]:
        """Get the current state of a component."""
        return self._states.get(component_id)
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Get a component by ID."""
        return self._components.get(component_id)
    
    async def initialize_component(self, component_id: str) -> bool:
        """Initialize a component."""
        component = self._components.get(component_id)
        if not component:
            self._logger.error(f"Component {component_id} not found")
            return False
        
        current_state = self._states.get(component_id)
        if current_state != ComponentState.CREATED:
            self._logger.warning(f"Component {component_id} is not in CREATED state")
            return False
        
        try:
            self._transition_state(component_id, ComponentState.INITIALIZING)
            
            config = self._configs.get(component_id, {})
            await self._run_async_or_sync(component.initialize, config)
            
            self._transition_state(component_id, ComponentState.INITIALIZED)
            self._logger.info(f"Initialized component {component_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize component {component_id}: {e}")
            self._transition_state(component_id, ComponentState.ERROR)
            return False
    
    async def start_component(self, component_id: str) -> bool:
        """Start a component."""
        component = self._components.get(component_id)
        if not component:
            self._logger.error(f"Component {component_id} not found")
            return False
        
        current_state = self._states.get(component_id)
        if current_state not in [ComponentState.INITIALIZED, ComponentState.STOPPED]:
            self._logger.warning(f"Component {component_id} is not in a startable state")
            return False
        
        try:
            self._transition_state(component_id, ComponentState.STARTING)
            
            await self._run_async_or_sync(component.start)
            
            self._transition_state(component_id, ComponentState.RUNNING)
            self._logger.info(f"Started component {component_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start component {component_id}: {e}")
            self._transition_state(component_id, ComponentState.ERROR)
            return False
    
    async def stop_component(self, component_id: str) -> bool:
        """Stop a component."""
        component = self._components.get(component_id)
        if not component:
            self._logger.error(f"Component {component_id} not found")
            return False
        
        current_state = self._states.get(component_id)
        if current_state != ComponentState.RUNNING:
            self._logger.warning(f"Component {component_id} is not running")
            return False
        
        try:
            self._transition_state(component_id, ComponentState.STOPPING)
            
            await self._run_async_or_sync(component.stop)
            
            self._transition_state(component_id, ComponentState.STOPPED)
            self._logger.info(f"Stopped component {component_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to stop component {component_id}: {e}")
            self._transition_state(component_id, ComponentState.ERROR)
            return False
    
    async def restart_component(self, component_id: str) -> bool:
        """Restart a component."""
        if not await self.stop_component(component_id):
            return False
        
        # Wait a bit before restarting
        await asyncio.sleep(1)
        
        return await self.start_component(component_id)
    
    async def health_check_component(self, component_id: str) -> bool:
        """Perform a health check on a component."""
        component = self._components.get(component_id)
        if not component:
            return False
        
        try:
            is_healthy = await self._run_async_or_sync(component.health_check)
            if not is_healthy:
                self._logger.warning(f"Component {component_id} failed health check")
            return is_healthy
            
        except Exception as e:
            self._logger.error(f"Health check failed for component {component_id}: {e}")
            return False
    
    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all registered components."""
        results = {}
        for component_id in self._components:
            results[component_id] = await self.initialize_component(component_id)
        return results
    
    async def start_all(self) -> Dict[str, bool]:
        """Start all registered components."""
        results = {}
        for component_id in self._components:
            results[component_id] = await self.start_component(component_id)
        return results
    
    async def stop_all(self) -> Dict[str, bool]:
        """Stop all registered components."""
        results = {}
        for component_id in self._components:
            results[component_id] = await self.stop_component(component_id)
        return results
    
    def add_event_handler(self, event_type: str, handler: Callable[[LifecycleEvent], None]) -> None:
        """Add an event handler for lifecycle events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def get_component_info(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a component."""
        component = self._components.get(component_id)
        if not component:
            return None
        
        return {
            "component_id": component_id,
            "name": component.name,
            "type": component.component_type.value,
            "state": self._states.get(component_id).value,
            "created_at": component.created_at.isoformat(),
            "config": self._configs.get(component_id, {})
        }
    
    def list_components(self, state: ComponentState = None) -> List[str]:
        """List all components, optionally filtered by state."""
        if state is None:
            return list(self._components.keys())
        else:
            return [comp_id for comp_id, comp_state in self._states.items() 
                   if comp_state == state]
    
    def _transition_state(self, component_id: str, new_state: ComponentState) -> None:
        """Transition a component to a new state."""
        old_state = self._states.get(component_id)
        self._states[component_id] = new_state
        
        # Fire lifecycle event
        event = LifecycleEvent(component_id, "state_transition", old_state, new_state)
        self._fire_event("state_transition", event)
        
        self._logger.debug(f"Component {component_id} transitioned from {old_state} to {new_state}")
    
    def _fire_event(self, event_type: str, event: LifecycleEvent) -> None:
        """Fire a lifecycle event."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self._logger.error(f"Error in event handler: {e}")
    
    async def _run_async_or_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Run a function whether it's async or sync."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)


# Global lifecycle manager instance
_global_lifecycle_manager = ComponentLifecycleManager()


def get_global_lifecycle_manager() -> ComponentLifecycleManager:
    """Get the global lifecycle manager."""
    return _global_lifecycle_manager
