"""
LLMFlow Conductor Manager

This module implements the conductor management system for monitoring
and managing molecules and cells in the LLMFlow system.
"""

import asyncio
import uuid
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time

from ..core.base import ServiceAtom, Component, ComponentType
from ..queue import QueueManager, QueueClient
from ..molecules.optimization import PerformanceMetrics, PerformanceMetricsAtom

logger = logging.getLogger(__name__)


class ProcessStatus(Enum):
    """Status of a managed process."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RESTARTING = "restarting"


@dataclass
class ProcessInfo:
    """Information about a managed process."""
    process_id: str
    component_name: str
    component_type: ComponentType
    queue_bindings: List[str] = field(default_factory=list)
    status: ProcessStatus = ProcessStatus.STOPPED
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    restart_count: int = 0
    max_restarts: int = 3
    health_check_interval: float = 30.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if the process is healthy."""
        if self.status != ProcessStatus.RUNNING:
            return False
        
        if not self.last_health_check:
            return False
        
        # Check if health check is recent
        time_since_check = datetime.utcnow() - self.last_health_check
        return time_since_check < timedelta(seconds=self.health_check_interval * 2)
    
    def should_restart(self) -> bool:
        """Check if the process should be restarted."""
        return (self.status == ProcessStatus.FAILED and 
                self.restart_count < self.max_restarts)


class ConductorManager:
    """Central conductor management system."""
    
    def __init__(self, queue_manager: QueueManager):
        self.queue_manager = queue_manager
        self.conductor_id = str(uuid.uuid4())
        self.managed_processes: Dict[str, ProcessInfo] = {}
        self.running = False
        
        # Management tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_collection_task: Optional[asyncio.Task] = None
        self._restart_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.config = {
            'health_check_interval': 30.0,
            'metrics_collection_interval': 60.0,
            'restart_check_interval': 10.0,
            'max_concurrent_restarts': 3,
            'process_startup_timeout': 60.0,
            'process_shutdown_timeout': 30.0
        }
        
        # Metrics
        self.conductor_metrics = {
            'managed_processes': 0,
            'running_processes': 0,
            'failed_processes': 0,
            'total_restarts': 0,
            'health_checks_performed': 0,
            'metrics_collections': 0
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    async def start(self) -> None:
        """Start the conductor manager."""
        if self.running:
            return
        
        self.running = True
        
        # Start management tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
        self._restart_task = asyncio.create_task(self._restart_loop())
        
        logger.info(f"Conductor manager {self.conductor_id} started")
    
    async def stop(self) -> None:
        """Stop the conductor manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel management tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._metrics_collection_task:
            self._metrics_collection_task.cancel()
            try:
                await self._metrics_collection_task
            except asyncio.CancelledError:
                pass
        
        if self._restart_task:
            self._restart_task.cancel()
            try:
                await self._restart_task
            except asyncio.CancelledError:
                pass
        
        # Stop all managed processes
        await self._stop_all_processes()
        
        logger.info(f"Conductor manager {self.conductor_id} stopped")
    
    async def register_process(self, component: Component, 
                             queue_bindings: List[str] = None) -> str:
        """Register a process for management."""
        process_id = str(uuid.uuid4())
        
        process_info = ProcessInfo(
            process_id=process_id,
            component_name=component.name,
            component_type=component.component_type,
            queue_bindings=queue_bindings or []
        )
        
        self.managed_processes[process_id] = process_info
        self.conductor_metrics['managed_processes'] += 1
        
        # Emit event
        await self._emit_event('process_registered', {
            'process_id': process_id,
            'component_name': component.name,
            'component_type': component.component_type.value
        })
        
        logger.info(f"Registered process {process_id} for component {component.name}")
        return process_id
    
    async def start_process(self, process_id: str) -> bool:
        """Start a managed process."""
        if process_id not in self.managed_processes:
            logger.error(f"Process {process_id} not found")
            return False
        
        process_info = self.managed_processes[process_id]
        
        if process_info.status == ProcessStatus.RUNNING:
            logger.warning(f"Process {process_id} is already running")
            return True
        
        try:
            process_info.status = ProcessStatus.STARTING
            process_info.started_at = datetime.utcnow()
            
            # In a real implementation, this would actually start the process
            # For now, we'll simulate process startup
            await asyncio.sleep(0.1)  # Simulate startup time
            
            # Simulate getting PID
            process_info.pid = 12345  # Mock PID
            process_info.status = ProcessStatus.RUNNING
            process_info.last_health_check = datetime.utcnow()
            
            self.conductor_metrics['running_processes'] += 1
            
            # Emit event
            await self._emit_event('process_started', {
                'process_id': process_id,
                'component_name': process_info.component_name,
                'pid': process_info.pid
            })
            
            logger.info(f"Started process {process_id} (PID: {process_info.pid})")
            return True
            
        except Exception as e:
            process_info.status = ProcessStatus.FAILED
            self.conductor_metrics['failed_processes'] += 1
            
            logger.error(f"Failed to start process {process_id}: {e}")
            return False
    
    async def stop_process(self, process_id: str) -> bool:
        """Stop a managed process."""
        if process_id not in self.managed_processes:
            logger.error(f"Process {process_id} not found")
            return False
        
        process_info = self.managed_processes[process_id]
        
        if process_info.status == ProcessStatus.STOPPED:
            logger.warning(f"Process {process_id} is already stopped")
            return True
        
        try:
            process_info.status = ProcessStatus.STOPPING
            
            # In a real implementation, this would actually stop the process
            # For now, we'll simulate process shutdown
            await asyncio.sleep(0.1)  # Simulate shutdown time
            
            process_info.status = ProcessStatus.STOPPED
            process_info.pid = None
            
            if self.conductor_metrics['running_processes'] > 0:
                self.conductor_metrics['running_processes'] -= 1
            
            # Emit event
            await self._emit_event('process_stopped', {
                'process_id': process_id,
                'component_name': process_info.component_name
            })
            
            logger.info(f"Stopped process {process_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop process {process_id}: {e}")
            return False
    
    async def restart_process(self, process_id: str) -> bool:
        """Restart a managed process."""
        if process_id not in self.managed_processes:
            logger.error(f"Process {process_id} not found")
            return False
        
        process_info = self.managed_processes[process_id]
        
        if process_info.restart_count >= process_info.max_restarts:
            logger.warning(f"Process {process_id} has exceeded max restarts")
            return False
        
        try:
            process_info.status = ProcessStatus.RESTARTING
            process_info.restart_count += 1
            self.conductor_metrics['total_restarts'] += 1
            
            # Stop the process first
            await self.stop_process(process_id)
            
            # Wait a bit before restarting
            await asyncio.sleep(1.0)
            
            # Start the process again
            success = await self.start_process(process_id)
            
            if success:
                logger.info(f"Restarted process {process_id}")
            else:
                process_info.status = ProcessStatus.FAILED
                self.conductor_metrics['failed_processes'] += 1
                logger.error(f"Failed to restart process {process_id}")
            
            return success
            
        except Exception as e:
            process_info.status = ProcessStatus.FAILED
            self.conductor_metrics['failed_processes'] += 1
            logger.error(f"Error restarting process {process_id}: {e}")
            return False
    
    async def get_process_metrics(self, process_id: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a process."""
        if process_id not in self.managed_processes:
            return None
        
        process_info = self.managed_processes[process_id]
        
        if process_info.status != ProcessStatus.RUNNING:
            return None
        
        # In a real implementation, this would collect actual metrics
        # For now, we'll return simulated metrics
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            latency_ms=50.0,
            throughput_ops_per_sec=1000.0,
            error_rate=0.01,
            memory_usage_mb=128.0,
            cpu_usage_percent=25.0,
            queue_depth=10,
            processed_messages=1000
        )
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['health_check_interval'])
                
                for process_id, process_info in self.managed_processes.items():
                    if process_info.status == ProcessStatus.RUNNING:
                        await self._perform_health_check(process_id)
                
                self.conductor_metrics['health_checks_performed'] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['metrics_collection_interval'])
                
                for process_id, process_info in self.managed_processes.items():
                    if process_info.status == ProcessStatus.RUNNING:
                        await self._collect_process_metrics(process_id)
                
                self.conductor_metrics['metrics_collections'] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
    
    async def _restart_loop(self) -> None:
        """Background restart loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['restart_check_interval'])
                
                # Check for processes that need restarting
                processes_to_restart = []
                for process_id, process_info in self.managed_processes.items():
                    if process_info.should_restart():
                        processes_to_restart.append(process_id)
                
                # Restart processes (with concurrency limit)
                concurrent_restarts = 0
                for process_id in processes_to_restart:
                    if concurrent_restarts >= self.config['max_concurrent_restarts']:
                        break
                    
                    asyncio.create_task(self.restart_process(process_id))
                    concurrent_restarts += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in restart loop: {e}")
    
    async def _perform_health_check(self, process_id: str) -> None:
        """Perform health check on a process."""
        process_info = self.managed_processes[process_id]
        
        try:
            # In a real implementation, this would check process health
            # For now, we'll simulate health check
            await asyncio.sleep(0.01)  # Simulate health check time
            
            # Update last health check time
            process_info.last_health_check = datetime.utcnow()
            
            # Simulate occasional health check failures
            import random
            if random.random() < 0.01:  # 1% chance of failure
                process_info.status = ProcessStatus.FAILED
                logger.warning(f"Health check failed for process {process_id}")
            
        except Exception as e:
            process_info.status = ProcessStatus.FAILED
            logger.error(f"Health check error for process {process_id}: {e}")
    
    async def _collect_process_metrics(self, process_id: str) -> None:
        """Collect metrics for a process."""
        process_info = self.managed_processes[process_id]
        
        try:
            # Get performance metrics
            metrics = await self.get_process_metrics(process_id)
            
            if metrics:
                # Store metrics in process info
                process_info.metrics = {
                    'timestamp': metrics.timestamp.isoformat(),
                    'latency_ms': metrics.latency_ms,
                    'throughput_ops_per_sec': metrics.throughput_ops_per_sec,
                    'error_rate': metrics.error_rate,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'cpu_usage_percent': metrics.cpu_usage_percent,
                    'queue_depth': metrics.queue_depth,
                    'processed_messages': metrics.processed_messages
                }
                
                # Send metrics to master queue for analysis
                await self._send_metrics_to_master(process_id, metrics)
            
        except Exception as e:
            logger.error(f"Error collecting metrics for process {process_id}: {e}")
    
    async def _send_metrics_to_master(self, process_id: str, metrics: PerformanceMetrics) -> None:
        """Send metrics to master queue for analysis."""
        try:
            # Create metrics atom
            metrics_atom = PerformanceMetricsAtom(metrics)
            
            # Send to master queue (system.metrics)
            await self.queue_manager.enqueue(
                'system.metrics',
                {
                    'process_id': process_id,
                    'conductor_id': self.conductor_id,
                    'metrics': metrics_atom.value
                },
                domain='system'
            )
            
        except Exception as e:
            logger.error(f"Error sending metrics to master: {e}")
    
    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Emit an event to all registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_type, event_data)
                    else:
                        handler(event_type, event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    async def _stop_all_processes(self) -> None:
        """Stop all managed processes."""
        for process_id in list(self.managed_processes.keys()):
            try:
                await self.stop_process(process_id)
            except Exception as e:
                logger.error(f"Error stopping process {process_id}: {e}")
