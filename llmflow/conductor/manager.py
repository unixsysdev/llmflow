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
    
    def __init__(self, queue_manager: QueueManager, security_enabled: bool = True):
        self.queue_manager = queue_manager
        self.security_enabled = security_enabled
        self.conductor_id = str(uuid.uuid4())
        self.managed_processes: Dict[str, ProcessInfo] = {}
        self.running = False
        
        # Security integration
        if security_enabled:
            try:
                from ..security import get_auth_manager, get_authorization_manager
                self.auth_manager = get_auth_manager()
                self.authz_manager = get_authorization_manager()
            except ImportError:
                logger.warning("Security modules not available, running without security")
                self.auth_manager = None
                self.authz_manager = None
                self.security_enabled = False
        else:
            self.auth_manager = None
            self.authz_manager = None
        
        # Management tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_collection_task: Optional[asyncio.Task] = None
        self._restart_task: Optional[asyncio.Task] = None
        self._security_audit_task: Optional[asyncio.Task] = None
        self._performance_analysis_task: Optional[asyncio.Task] = None
        
        # Enhanced configuration
        self.config = {
            'health_check_interval': 15.0,  # More frequent health checks
            'metrics_collection_interval': 30.0,  # More frequent metrics
            'restart_check_interval': 5.0,  # Faster restart detection
            'security_audit_interval': 300.0,  # Security audit every 5 minutes
            'performance_analysis_interval': 120.0,  # Performance analysis every 2 minutes
            'max_concurrent_restarts': 3,
            'process_startup_timeout': 60.0,
            'process_shutdown_timeout': 30.0,
            'anomaly_detection_enabled': True,
            'auto_scaling_enabled': True,
            'predictive_restart_enabled': True
        }
        
        # Enhanced metrics with security tracking
        self.conductor_metrics = {
            'managed_processes': 0,
            'running_processes': 0,
            'failed_processes': 0,
            'total_restarts': 0,
            'health_checks_performed': 0,
            'metrics_collections': 0,
            'security_violations': 0,
            'performance_anomalies': 0,
            'predictive_restarts': 0,
            'auto_scale_events': 0
        }
        
        # Performance analytics
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.anomaly_thresholds = {
            'latency_multiplier': 3.0,
            'error_rate_threshold': 0.1,
            'memory_growth_rate': 0.2,
            'cpu_spike_threshold': 0.9
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
            
            # Start enhanced monitoring tasks
            if self.security_enabled:
                self._security_audit_task = asyncio.create_task(self._security_audit_loop())
            
            self._performance_analysis_task = asyncio.create_task(self._performance_analysis_loop())
            
            logger.info(f"Conductor manager {self.conductor_id} started")

    
    async def stop(self) -> None:
            """Stop the conductor manager."""
            if not self.running:
                return
            
            self.running = False
            
            # Cancel management tasks
            tasks_to_cancel = [
                self._health_check_task,
                self._metrics_collection_task,
                self._restart_task,
                self._security_audit_task,
                self._performance_analysis_task
            ]
            
            for task in tasks_to_cancel:
                if task:
                    task.cancel()
                    try:
                        await task
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
    
        async def _security_audit_loop(self) -> None:
            """Security audit loop for enhanced monitoring."""
            if not self.security_enabled:
                return
                
            logger.info("Security audit loop started")
            
            while self.running:
                try:
                    await asyncio.sleep(self.config['security_audit_interval'])
                    
                    # Perform security audit
                    await self._perform_security_audit()
                    
                    self.conductor_metrics['security_audits_performed'] = \
                        self.conductor_metrics.get('security_audits_performed', 0) + 1
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in security audit loop: {e}")
            
            logger.info("Security audit loop stopped")
        
        async def _performance_analysis_loop(self) -> None:
            """Performance analysis loop for enhanced monitoring."""
            logger.info("Performance analysis loop started")
            
            while self.running:
                try:
                    await asyncio.sleep(self.config['performance_analysis_interval'])
                    
                    # Analyze performance
                    await self._analyze_performance()
                    
                    self.conductor_metrics['performance_analyses'] = \
                        self.conductor_metrics.get('performance_analyses', 0) + 1
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in performance analysis loop: {e}")
            
            logger.info("Performance analysis loop stopped")
        
        async def _analyze_performance(self) -> None:
            """Analyze performance metrics for anomalies and optimization opportunities."""
            try:
                current_time = datetime.utcnow()
                
                for process_id, process_info in self.managed_processes.items():
                    if process_info.status != ProcessStatus.RUNNING:
                        continue
                    
                    # Get recent metrics
                    if process_id not in self.performance_history:
                        self.performance_history[process_id] = []
                    
                    recent_metrics = self.performance_history[process_id][-10:]  # Last 10 measurements
                    
                    if len(recent_metrics) < 3:
                        continue  # Need at least 3 measurements for analysis
                    
                    # Detect anomalies
                    anomalies = await self._detect_performance_anomalies(process_id, recent_metrics)
                    
                    if anomalies:
                        logger.warning(f"Performance anomalies detected for {process_id}: {anomalies}")
                        
                        # Request optimization if needed
                        if any(severity == 'high' for _, severity in anomalies):
                            await self._request_optimization(process_id, anomalies)
                        
                        self.conductor_metrics['performance_anomalies'] = \
                            self.conductor_metrics.get('performance_anomalies', 0) + len(anomalies)
                    
                    # Check for predictive restart conditions
                    if self.config.get('predictive_restart_enabled', False):
                        should_restart = await self._predict_restart_need(process_id, recent_metrics)
                        if should_restart:
                            logger.info(f"Predictive restart triggered for {process_id}")
                            await self._perform_predictive_restart(process_id)
            
            except Exception as e:
                logger.error(f"Error analyzing performance: {e}")
        
        async def _request_optimization(self, process_id: str, anomalies: List[Tuple[str, str]]) -> None:
            """Request optimization for a process with performance issues."""
            try:
                process_info = self.managed_processes.get(process_id)
                if not process_info:
                    return
                
                # Prepare optimization request
                optimization_request = {
                    'conductor_id': self.conductor_id,
                    'process_id': process_id,
                    'component_name': process_info.component.name if hasattr(process_info.component, 'name') else 'unknown',
                    'component_type': process_info.component_type.value,
                    'anomalies': [{'type': anom_type, 'severity': severity} for anom_type, severity in anomalies],
                    'performance_history': self.performance_history.get(process_id, [])[-5:],  # Last 5 measurements
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Send to master queue optimizer
                await self.queue_manager.enqueue(
                    'system.optimization_requests',
                    optimization_request,
                    domain='system'
                )
                
                logger.info(f"Optimization request sent for {process_id}")
                
            except Exception as e:
                logger.error(f"Error requesting optimization for {process_id}: {e}")
        
        async def _allocate_resources(self, process_id: str, resource_requirements: Dict[str, Any]) -> bool:
            """Allocate resources for a process."""
            try:
                process_info = self.managed_processes.get(process_id)
                if not process_info:
                    return False
                
                # Simulate resource allocation
                logger.info(f"Allocating resources for {process_id}: {resource_requirements}")
                
                # Update process info with allocated resources
                if not hasattr(process_info, 'allocated_resources'):
                    process_info.allocated_resources = {}
                
                process_info.allocated_resources.update(resource_requirements)
                
                # Track resource allocation in metrics
                self.conductor_metrics['resource_allocations'] = \
                    self.conductor_metrics.get('resource_allocations', 0) + 1
                
                return True
                
            except Exception as e:
                logger.error(f"Error allocating resources for {process_id}: {e}")
                return False
        
        async def _cleanup_resources(self, process_id: str) -> None:
            """Clean up resources allocated to a process."""
            try:
                process_info = self.managed_processes.get(process_id)
                if not process_info:
                    return
                
                # Clean up allocated resources
                if hasattr(process_info, 'allocated_resources'):
                    logger.info(f"Cleaning up resources for {process_id}: {process_info.allocated_resources}")
                    process_info.allocated_resources.clear()
                
                # Track resource cleanup in metrics
                self.conductor_metrics['resource_cleanups'] = \
                    self.conductor_metrics.get('resource_cleanups', 0) + 1
                
            except Exception as e:
                logger.error(f"Error cleaning up resources for {process_id}: {e}")    
        
            async def _detect_performance_anomalies(self, process_id: str, recent_metrics: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
                """Detect performance anomalies in recent metrics."""
                anomalies = []
                
                try:
                    if len(recent_metrics) < 3:
                        return anomalies
                    
                    # Calculate averages and trends
                    current_metrics = recent_metrics[-1]
                    baseline_metrics = recent_metrics[:-1]
                    
                    # Latency anomaly detection
                    baseline_latency = sum(m.get('latency_ms', 0) for m in baseline_metrics) / len(baseline_metrics)
                    current_latency = current_metrics.get('latency_ms', 0)
                    
                    if current_latency > baseline_latency * self.anomaly_thresholds['latency_multiplier']:
                        severity = 'high' if current_latency > baseline_latency * 5 else 'medium'
                        anomalies.append(('latency_spike', severity))
                    
                    # Error rate anomaly detection
                    current_error_rate = current_metrics.get('error_rate', 0)
                    if current_error_rate > self.anomaly_thresholds['error_rate_threshold']:
                        severity = 'high' if current_error_rate > 0.1 else 'medium'
                        anomalies.append(('high_error_rate', severity))
                    
                    # Memory growth anomaly detection
                    if len(recent_metrics) >= 5:
                        memory_values = [m.get('memory_usage_mb', 0) for m in recent_metrics]
                        memory_growth_rate = self._calculate_growth_rate(memory_values)
                        
                        if memory_growth_rate > self.anomaly_thresholds['memory_growth_rate']:
                            severity = 'high' if memory_growth_rate > 0.5 else 'medium'
                            anomalies.append(('memory_leak', severity))
                    
                    # CPU spike detection
                    current_cpu = current_metrics.get('cpu_usage', 0)
                    if current_cpu > self.anomaly_thresholds['cpu_spike_threshold']:
                        anomalies.append(('cpu_spike', 'medium'))
                
                except Exception as e:
                    logger.error(f"Error detecting anomalies for {process_id}: {e}")
                
                return anomalies
            
            async def _predict_restart_need(self, process_id: str, recent_metrics: List[Dict[str, Any]]) -> bool:
                """Predict if a process needs restart based on performance trends."""
                try:
                    if len(recent_metrics) < 5:
                        return False
                    
                    # Check for degrading performance trends
                    latency_values = [m.get('latency_ms', 0) for m in recent_metrics]
                    memory_values = [m.get('memory_usage_mb', 0) for m in recent_metrics]
                    error_values = [m.get('error_rate', 0) for m in recent_metrics]
                    
                    # Calculate trends
                    latency_trend = self._calculate_growth_rate(latency_values)
                    memory_trend = self._calculate_growth_rate(memory_values)
                    error_trend = self._calculate_growth_rate(error_values)
                    
                    # Predict restart need based on trends
                    restart_indicators = 0
                    
                    if latency_trend > 0.2:  # 20% increase in latency
                        restart_indicators += 1
                    
                    if memory_trend > 0.1:  # 10% increase in memory
                        restart_indicators += 1
                    
                    if error_trend > 0.05:  # 5% increase in errors
                        restart_indicators += 2  # Errors are more critical
                    
                    # Restart if multiple indicators suggest degradation
                    return restart_indicators >= 2
                
                except Exception as e:
                    logger.error(f"Error predicting restart need for {process_id}: {e}")
                    return False
            
            async def _perform_predictive_restart(self, process_id: str) -> None:
                """Perform a predictive restart of a process."""
                try:
                    logger.info(f"Performing predictive restart for {process_id}")
                    
                    # Mark as predictive restart
                    self.conductor_metrics['predictive_restarts'] = \
                        self.conductor_metrics.get('predictive_restarts', 0) + 1
                    
                    # Restart the process
                    success = await self.restart_process(process_id)
                    
                    if success:
                        logger.info(f"Predictive restart successful for {process_id}")
                    else:
                        logger.error(f"Predictive restart failed for {process_id}")
                
                except Exception as e:
                    logger.error(f"Error performing predictive restart for {process_id}: {e}")
            
            async def _perform_security_audit(self) -> None:
                """Perform security audit of managed processes."""
                try:
                    if not self.security_enabled or not self.auth_manager:
                        return
                    
                    audit_results = []
                    
                    for process_id, process_info in self.managed_processes.items():
                        # Check process authentication status
                        auth_status = await self._check_process_auth(process_id)
                        
                        # Check for security violations
                        violations = await self._check_security_violations(process_id)
                        
                        if violations:
                            audit_results.append({
                                'process_id': process_id,
                                'violations': violations,
                                'timestamp': datetime.utcnow().isoformat()
                            })
                            
                            self.conductor_metrics['security_violations'] = \
                                self.conductor_metrics.get('security_violations', 0) + len(violations)
                    
                    # Send audit results if any violations found
                    if audit_results:
                        await self.queue_manager.enqueue(
                            'system.security_audit',
                            {
                                'conductor_id': self.conductor_id,
                                'audit_results': audit_results,
                                'timestamp': datetime.utcnow().isoformat()
                            },
                            domain='system'
                        )
                
                except Exception as e:
                    logger.error(f"Error performing security audit: {e}")
            
            async def _check_process_auth(self, process_id: str) -> Dict[str, Any]:
                """Check authentication status of a process."""
                try:
                    # Simulate authentication check
                    return {
                        'authenticated': True,
                        'token_valid': True,
                        'last_check': datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Error checking auth for {process_id}: {e}")
                    return {'authenticated': False, 'error': str(e)}
            
            async def _check_security_violations(self, process_id: str) -> List[str]:
                """Check for security violations in a process."""
                violations = []
                
                try:
                    process_info = self.managed_processes.get(process_id)
                    if not process_info:
                        return violations
                    
                    # Check for common security issues
                    # This is a simplified example - real implementation would be more comprehensive
                    
                    # Check if process has been running too long without restart
                    if process_info.start_time:
                        uptime = datetime.utcnow() - process_info.start_time
                        if uptime.total_seconds() > 86400:  # 24 hours
                            violations.append('extended_uptime')
                    
                    # Check restart count
                    if process_info.restart_count > 10:
                        violations.append('excessive_restarts')
                
                except Exception as e:
                    logger.error(f"Error checking security violations for {process_id}: {e}")
                
                return violations
            
            def _calculate_growth_rate(self, values: List[float]) -> float:
                """Calculate growth rate of a series of values."""
                if len(values) < 2:
                    return 0.0
                
                try:
                    # Simple linear growth rate calculation
                    start_value = values[0] if values[0] > 0 else 1
                    end_value = values[-1]
                    
                    growth_rate = (end_value - start_value) / start_value
                    return growth_rate
                
                except (ZeroDivisionError, TypeError):
                    return 0.0
    
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
                    
                    # Update performance history for analysis
                    if process_id not in self.performance_history:
                        self.performance_history[process_id] = []
                    
                    # Add to performance history (keep last 50 measurements)
                    self.performance_history[process_id].append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'latency_ms': metrics.latency_ms,
                        'throughput_ops_per_sec': metrics.throughput_ops_per_sec,
                        'error_rate': metrics.error_rate,
                        'memory_usage_mb': metrics.memory_usage_mb,
                        'cpu_usage': metrics.cpu_usage_percent / 100.0,  # Convert to 0-1 range
                        'queue_depth': metrics.queue_depth
                    })
                    
                    # Keep only recent history
                    if len(self.performance_history[process_id]) > 50:
                        self.performance_history[process_id] = self.performance_history[process_id][-50:]
                    
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
        """Stop all managed processes with enhanced coordination."""
        if not self.managed_processes:
            return
        
        logger.info(f"Stopping {len(self.managed_processes)} managed processes")
        
        # Group processes by priority (stop in reverse dependency order)
        processes_by_priority = {
            ComponentType.CELL: [],
            ComponentType.MOLECULE: [],
            ComponentType.ATOM: []
        }
        
        for process_id, process_info in self.managed_processes.items():
            if process_info.status == ProcessStatus.RUNNING:
                processes_by_priority[process_info.component_type].append(process_id)
        
        # Stop in order: Cells -> Molecules -> Atoms
        for component_type in [ComponentType.CELL, ComponentType.MOLECULE, ComponentType.ATOM]:
            if processes_by_priority[component_type]:
                logger.info(f"Stopping {len(processes_by_priority[component_type])} {component_type.value}s")
                
                # Stop processes in parallel within each tier
                stop_tasks = []
                for process_id in processes_by_priority[component_type]:
                    stop_tasks.append(self.stop_process(process_id))
                
                # Wait for all in this tier to stop
                await asyncio.gather(*stop_tasks, return_exceptions=True)
                
                # Small delay between tiers
                await asyncio.sleep(0.5)
        
        logger.info("All managed processes stopped")
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler for enhanced monitoring."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """Remove an event handler."""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def get_conductor_status(self) -> Dict[str, Any]:
        """Get comprehensive conductor status."""
        return {
            'conductor_id': self.conductor_id,
            'running': self.running,
            'security_enabled': self.security_enabled,
            'managed_processes': len(self.managed_processes),
            'metrics': self.conductor_metrics.copy(),
            'config': self.config.copy(),
            'process_statuses': {
                status.value: len([p for p in self.managed_processes.values() if p.status == status])
                for status in ProcessStatus
            }
        }
