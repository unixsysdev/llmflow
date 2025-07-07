"""
LLMFlow Master Queue Analytics

This module implements the analytics system for performance monitoring
and system-wide insights in LLMFlow.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import uuid

from ..queue import QueueManager
from ..molecules.optimization import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class SystemSnapshot:
    """A snapshot of system performance at a point in time."""
    timestamp: datetime
    total_components: int
    active_components: int
    total_queues: int
    total_messages: int
    avg_latency_ms: float
    avg_throughput_ops_per_sec: float
    avg_error_rate: float
    avg_memory_usage_mb: float
    avg_cpu_usage_percent: float
    total_queue_depth: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_components': self.total_components,
            'active_components': self.active_components,
            'total_queues': self.total_queues,
            'total_messages': self.total_messages,
            'avg_latency_ms': self.avg_latency_ms,
            'avg_throughput_ops_per_sec': self.avg_throughput_ops_per_sec,
            'avg_error_rate': self.avg_error_rate,
            'avg_memory_usage_mb': self.avg_memory_usage_mb,
            'avg_cpu_usage_percent': self.avg_cpu_usage_percent,
            'total_queue_depth': self.total_queue_depth
        }


@dataclass
class ComponentAnalytics:
    """Analytics data for a component."""
    component_id: str
    component_name: str
    component_type: str
    metrics_count: int
    first_seen: datetime
    last_seen: datetime
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_throughput_ops_per_sec: float
    avg_error_rate: float
    total_errors: int
    avg_memory_usage_mb: float
    max_memory_usage_mb: float
    avg_cpu_usage_percent: float
    max_cpu_usage_percent: float
    optimization_count: int
    last_optimization: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component_id': self.component_id,
            'component_name': self.component_name,
            'component_type': self.component_type,
            'metrics_count': self.metrics_count,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'avg_latency_ms': self.avg_latency_ms,
            'min_latency_ms': self.min_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'avg_throughput_ops_per_sec': self.avg_throughput_ops_per_sec,
            'avg_error_rate': self.avg_error_rate,
            'total_errors': self.total_errors,
            'avg_memory_usage_mb': self.avg_memory_usage_mb,
            'max_memory_usage_mb': self.max_memory_usage_mb,
            'avg_cpu_usage_percent': self.avg_cpu_usage_percent,
            'max_cpu_usage_percent': self.max_cpu_usage_percent,
            'optimization_count': self.optimization_count,
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None
        }


class PerformanceAnalytics:
    """Advanced performance analytics system."""
    
    def __init__(self, queue_manager: QueueManager):
        self.queue_manager = queue_manager
        self.running = False
        
        # Data storage
        self.component_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.system_snapshots: deque = deque(maxlen=10000)
        self.component_analytics: Dict[str, ComponentAnalytics] = {}
        
        # Configuration
        self.config = {
            'analytics_interval': 60.0,  # 1 minute
            'snapshot_interval': 300.0,  # 5 minutes
            'cleanup_interval': 86400.0,  # 24 hours
            'metrics_retention_days': 30,
            'snapshot_retention_days': 7
        }
        
        # Background tasks
        self._analytics_task: Optional[asyncio.Task] = None
        self._snapshot_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the analytics system."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self._analytics_task = asyncio.create_task(self._analytics_loop())
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Performance analytics system started")
    
    async def stop(self) -> None:
        """Stop the analytics system."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        for task in [self._analytics_task, self._snapshot_task, self._cleanup_task]:
            if task:
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            *[task for task in [self._analytics_task, self._snapshot_task, self._cleanup_task] if task],
            return_exceptions=True
        )
        
        logger.info("Performance analytics system stopped")
    
    async def record_metrics(self, component_id: str, metrics: PerformanceMetrics) -> None:
        """Record metrics for a component."""
        self.component_metrics[component_id].append(metrics)
        
        # Keep only recent metrics
        retention_cutoff = datetime.utcnow() - timedelta(days=self.config['metrics_retention_days'])
        self.component_metrics[component_id] = [
            m for m in self.component_metrics[component_id]
            if m.timestamp >= retention_cutoff
        ]
        
        # Update component analytics
        await self._update_component_analytics(component_id)
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide analytics overview."""
        if not self.system_snapshots:
            return {}
        
        latest_snapshot = self.system_snapshots[-1]
        
        return {
            'latest_snapshot': latest_snapshot.to_dict(),
            'total_components': len(self.component_analytics),
            'active_components': len([c for c in self.component_analytics.values() 
                                    if (datetime.utcnow() - c.last_seen).total_seconds() < 300]),
            'total_snapshots': len(self.system_snapshots),
            'analytics_running': self.running
        }
    
    async def _analytics_loop(self) -> None:
        """Background analytics processing loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['analytics_interval'])
                
                # Process incoming metrics
                await self._process_metrics_queue()
                
                # Update all component analytics
                for component_id in self.component_metrics.keys():
                    await self._update_component_analytics(component_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}")
    
    async def _snapshot_loop(self) -> None:
        """Background system snapshot loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['snapshot_interval'])
                await self._create_system_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in snapshot loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['cleanup_interval'])
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _process_metrics_queue(self) -> None:
        """Process metrics from the system metrics queue."""
        try:
            metrics_data = await self.queue_manager.dequeue('system.metrics')
            
            if metrics_data:
                component_id = metrics_data.get('process_id')
                metrics_dict = metrics_data.get('metrics', {})
                
                if component_id and metrics_dict:
                    metrics = PerformanceMetrics(
                        timestamp=datetime.fromisoformat(metrics_dict['timestamp']),
                        latency_ms=metrics_dict['latency_ms'],
                        throughput_ops_per_sec=metrics_dict['throughput_ops_per_sec'],
                        error_rate=metrics_dict['error_rate'],
                        memory_usage_mb=metrics_dict['memory_usage_mb'],
                        cpu_usage_percent=metrics_dict['cpu_usage_percent'],
                        queue_depth=metrics_dict['queue_depth'],
                        processed_messages=metrics_dict['processed_messages']
                    )
                    
                    await self.record_metrics(component_id, metrics)
        
        except Exception as e:
            logger.error(f"Error processing metrics queue: {e}")
    
    async def _update_component_analytics(self, component_id: str) -> None:
        """Update analytics for a component."""
        metrics_list = self.component_metrics.get(component_id, [])
        
        if not metrics_list:
            return
        
        # Calculate analytics
        latencies = [m.latency_ms for m in metrics_list]
        throughputs = [m.throughput_ops_per_sec for m in metrics_list]
        error_rates = [m.error_rate for m in metrics_list]
        memory_usages = [m.memory_usage_mb for m in metrics_list]
        cpu_usages = [m.cpu_usage_percent for m in metrics_list]
        
        # Create or update analytics
        analytics = ComponentAnalytics(
            component_id=component_id,
            component_name=f"component_{component_id[:8]}",
            component_type="unknown",
            metrics_count=len(metrics_list),
            first_seen=min(m.timestamp for m in metrics_list),
            last_seen=max(m.timestamp for m in metrics_list),
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            avg_throughput_ops_per_sec=statistics.mean(throughputs),
            avg_error_rate=statistics.mean(error_rates),
            total_errors=sum(1 for rate in error_rates if rate > 0),
            avg_memory_usage_mb=statistics.mean(memory_usages),
            max_memory_usage_mb=max(memory_usages),
            avg_cpu_usage_percent=statistics.mean(cpu_usages),
            max_cpu_usage_percent=max(cpu_usages),
            optimization_count=0,
            last_optimization=None
        )
        
        self.component_analytics[component_id] = analytics
    
    async def _create_system_snapshot(self) -> None:
        """Create a system-wide performance snapshot."""
        if not self.component_analytics:
            return
        
        all_analytics = list(self.component_analytics.values())
        active_threshold = datetime.utcnow() - timedelta(minutes=5)
        active_components = [a for a in all_analytics if a.last_seen >= active_threshold]
        
        if not active_components:
            return
        
        snapshot = SystemSnapshot(
            timestamp=datetime.utcnow(),
            total_components=len(all_analytics),
            active_components=len(active_components),
            total_queues=0,
            total_messages=0,
            avg_latency_ms=statistics.mean([a.avg_latency_ms for a in active_components]),
            avg_throughput_ops_per_sec=statistics.mean([a.avg_throughput_ops_per_sec for a in active_components]),
            avg_error_rate=statistics.mean([a.avg_error_rate for a in active_components]),
            avg_memory_usage_mb=statistics.mean([a.avg_memory_usage_mb for a in active_components]),
            avg_cpu_usage_percent=statistics.mean([a.avg_cpu_usage_percent for a in active_components]),
            total_queue_depth=0
        )
        
        self.system_snapshots.append(snapshot)
        
        # Send snapshot to analytics queue
        await self._send_snapshot_to_queue(snapshot)
    
    async def _send_snapshot_to_queue(self, snapshot: SystemSnapshot) -> None:
        """Send snapshot to analytics queue."""
        try:
            await self.queue_manager.enqueue(
                'system.analytics',
                {
                    'type': 'system_snapshot',
                    'data': snapshot.to_dict()
                },
                domain='system'
            )
        except Exception as e:
            logger.error(f"Error sending snapshot to queue: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old analytics data."""
        metrics_cutoff = datetime.utcnow() - timedelta(days=self.config['metrics_retention_days'])
        
        for component_id in list(self.component_metrics.keys()):
            self.component_metrics[component_id] = [
                m for m in self.component_metrics[component_id]
                if m.timestamp >= metrics_cutoff
            ]
            
            if not self.component_metrics[component_id]:
                del self.component_metrics[component_id]
        
        # Clean up old snapshots
        snapshot_cutoff = datetime.utcnow() - timedelta(days=self.config['snapshot_retention_days'])
        
        while self.system_snapshots and self.system_snapshots[0].timestamp < snapshot_cutoff:
            self.system_snapshots.popleft()
        
        logger.info("Cleaned up old analytics data")
    
    async def get_analytics_status(self) -> Dict[str, Any]:
        """Get analytics system status."""
        return {
            'running': self.running,
            'components_tracked': len(self.component_metrics),
            'analytics_computed': len(self.component_analytics),
            'system_snapshots': len(self.system_snapshots),
            'config': self.config.copy(),
            'timestamp': datetime.utcnow().isoformat()
        }
