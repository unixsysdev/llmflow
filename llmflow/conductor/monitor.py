"""
LLMFlow Conductor Monitor

This module implements performance monitoring and metrics collection
for the LLMFlow conductor system.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import statistics
import json

from ..queue import QueueManager
from ..molecules.optimization import PerformanceMetrics, PerformanceMetricsAtom

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Configuration for monitoring alerts."""
    rule_id: str
    name: str
    condition: str  # Python expression to evaluate
    threshold: float
    severity: str  # 'info', 'warning', 'error', 'critical'
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self) -> bool:
        """Check if alert should trigger (respecting cooldown)."""
        if not self.enabled:
            return False
        
        if self.last_triggered is None:
            return True
        
        time_since_last = datetime.utcnow() - self.last_triggered
        return time_since_last >= timedelta(minutes=self.cooldown_minutes)
    
    def trigger(self) -> None:
        """Mark alert as triggered."""
        self.last_triggered = datetime.utcnow()


@dataclass
class Alert:
    """A triggered alert."""
    alert_id: str
    rule_id: str
    severity: str
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_resolved(self) -> bool:
        """Check if alert is resolved."""
        return self.resolved_at is not None
    
    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved_at = datetime.utcnow()


class ConductorMonitor:
    """Performance monitoring and alerting system."""
    
    def __init__(self, queue_manager: QueueManager):
        self.queue_manager = queue_manager
        self.running = False
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=10000)
        self.aggregated_metrics: Dict[str, Any] = {}
        
        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Monitoring tasks
        self._aggregation_task: Optional[asyncio.Task] = None
        self._alert_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.config = {
            'aggregation_interval': 60.0,  # 1 minute
            'alert_check_interval': 10.0,  # 10 seconds
            'cleanup_interval': 3600.0,    # 1 hour
            'metrics_retention_hours': 24,
            'alert_retention_days': 7
        }
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Initialize default alert rules
        self._initialize_default_alerts()
    
    def _initialize_default_alerts(self) -> None:
        """Initialize default monitoring alerts."""
        default_rules = [
            AlertRule(
                rule_id='high_latency',
                name='High Latency',
                condition='metrics.latency_ms > 1000',
                threshold=1000.0,
                severity='warning'
            ),
            AlertRule(
                rule_id='low_throughput',
                name='Low Throughput',
                condition='metrics.throughput_ops_per_sec < 100',
                threshold=100.0,
                severity='warning'
            ),
            AlertRule(
                rule_id='high_error_rate',
                name='High Error Rate',
                condition='metrics.error_rate > 0.05',
                threshold=0.05,
                severity='error'
            ),
            AlertRule(
                rule_id='high_memory_usage',
                name='High Memory Usage',
                condition='metrics.memory_usage_mb > 1024',
                threshold=1024.0,
                severity='warning'
            ),
            AlertRule(
                rule_id='high_cpu_usage',
                name='High CPU Usage',
                condition='metrics.cpu_usage_percent > 80',
                threshold=80.0,
                severity='warning'
            ),
            AlertRule(
                rule_id='high_queue_depth',
                name='High Queue Depth',
                condition='metrics.queue_depth > 1000',
                threshold=1000.0,
                severity='error'
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    async def start(self) -> None:
        """Start the monitor."""
        if self.running:
            return
        
        self.running = True
        
        # Start monitoring tasks
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        self._alert_task = asyncio.create_task(self._alert_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Conductor monitor started")
    
    async def stop(self) -> None:
        """Stop the monitor."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel tasks
        if self._aggregation_task:
            self._aggregation_task.cancel()
        if self._alert_task:
            self._alert_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Wait for tasks to complete
        for task in [self._aggregation_task, self._alert_task, self._cleanup_task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Conductor monitor stopped")
    
    async def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        # Add to history
        self.metrics_history.append({
            'timestamp': metrics.timestamp,
            'metrics': metrics
        })
        
        # Check for immediate alerts
        await self._check_immediate_alerts(metrics)
    
    async def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current aggregated metrics."""
        return self.aggregated_metrics.copy() if self.aggregated_metrics else None
    
    async def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        history = []
        for record in self.metrics_history:
            if record['timestamp'] >= cutoff_time:
                history.append({
                    'timestamp': record['timestamp'].isoformat(),
                    'metrics': record['metrics'].to_dict()
                })
        
        return history
    
    async def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    async def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return list(self.alert_rules.values())
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    async def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if alert.triggered_at >= cutoff_time
        ]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            logger.info(f"Resolved alert: {alert_id}")
            return True
        
        return False
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable) -> bool:
        """Remove an alert callback."""
        try:
            self.alert_callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    async def _aggregation_loop(self) -> None:
        """Background metrics aggregation loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['aggregation_interval'])
                await self._aggregate_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
    
    async def _alert_loop(self) -> None:
        """Background alert checking loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['alert_check_interval'])
                await self._check_alerts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
    
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
    
    async def _aggregate_metrics(self) -> None:
        """Aggregate recent metrics."""
        if not self.metrics_history:
            return
        
        # Get recent metrics (last 5 minutes)
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        recent_metrics = [
            record['metrics'] for record in self.metrics_history
            if record['timestamp'] >= cutoff_time
        ]
        
        if not recent_metrics:
            return
        
        # Calculate aggregated values
        self.aggregated_metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'sample_count': len(recent_metrics),
            'avg_latency_ms': statistics.mean([m.latency_ms for m in recent_metrics]),
            'max_latency_ms': max([m.latency_ms for m in recent_metrics]),
            'min_latency_ms': min([m.latency_ms for m in recent_metrics]),
            'avg_throughput_ops_per_sec': statistics.mean([m.throughput_ops_per_sec for m in recent_metrics]),
            'max_throughput_ops_per_sec': max([m.throughput_ops_per_sec for m in recent_metrics]),
            'avg_error_rate': statistics.mean([m.error_rate for m in recent_metrics]),
            'max_error_rate': max([m.error_rate for m in recent_metrics]),
            'avg_memory_usage_mb': statistics.mean([m.memory_usage_mb for m in recent_metrics]),
            'max_memory_usage_mb': max([m.memory_usage_mb for m in recent_metrics]),
            'avg_cpu_usage_percent': statistics.mean([m.cpu_usage_percent for m in recent_metrics]),
            'max_cpu_usage_percent': max([m.cpu_usage_percent for m in recent_metrics]),
            'avg_queue_depth': statistics.mean([m.queue_depth for m in recent_metrics]),
            'max_queue_depth': max([m.queue_depth for m in recent_metrics]),
            'total_processed_messages': sum([m.processed_messages for m in recent_metrics])
        }
        
        # Send aggregated metrics to system queue
        await self._send_aggregated_metrics()
    
    async def _send_aggregated_metrics(self) -> None:
        """Send aggregated metrics to system queue."""
        try:
            await self.queue_manager.enqueue(
                'system.metrics',
                {
                    'type': 'aggregated_metrics',
                    'data': self.aggregated_metrics
                },
                domain='system'
            )
        except Exception as e:
            logger.error(f"Error sending aggregated metrics: {e}")
    
    async def _check_immediate_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for immediate alerts based on current metrics."""
        for rule in self.alert_rules.values():
            if not rule.enabled or not rule.should_trigger():
                continue
            
            # Evaluate condition
            if await self._evaluate_alert_condition(rule, metrics):
                await self._trigger_alert(rule, metrics)
    
    async def _check_alerts(self) -> None:
        """Check all alert conditions."""
        if not self.aggregated_metrics:
            return
        
        # Create a mock metrics object from aggregated data
        mock_metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            latency_ms=self.aggregated_metrics['avg_latency_ms'],
            throughput_ops_per_sec=self.aggregated_metrics['avg_throughput_ops_per_sec'],
            error_rate=self.aggregated_metrics['avg_error_rate'],
            memory_usage_mb=self.aggregated_metrics['avg_memory_usage_mb'],
            cpu_usage_percent=self.aggregated_metrics['avg_cpu_usage_percent'],
            queue_depth=self.aggregated_metrics['avg_queue_depth'],
            processed_messages=self.aggregated_metrics['total_processed_messages']
        )
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check if alert should trigger
            if await self._evaluate_alert_condition(rule, mock_metrics):
                if rule.should_trigger():
                    await self._trigger_alert(rule, mock_metrics)
            else:
                # Check if we should resolve existing alert
                await self._maybe_resolve_alert(rule.rule_id)
    
    async def _evaluate_alert_condition(self, rule: AlertRule, metrics: PerformanceMetrics) -> bool:
        """Evaluate alert condition."""
        try:
            # Create evaluation context
            context = {
                'metrics': metrics,
                'threshold': rule.threshold
            }
            
            # Evaluate condition
            return eval(rule.condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error(f"Error evaluating alert condition for {rule.rule_id}: {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, metrics: PerformanceMetrics) -> None:
        """Trigger an alert."""
        import uuid
        
        alert_id = str(uuid.uuid4())
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            severity=rule.severity,
            message=f"{rule.name}: {rule.condition}",
            triggered_at=datetime.utcnow(),
            metadata={
                'metrics': metrics.to_dict(),
                'threshold': rule.threshold
            }
        )
        
        # Add to active alerts
        self.active_alerts[alert_id] = alert
        
        # Mark rule as triggered
        rule.trigger()
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Send alert to system queue
        await self._send_alert_notification(alert)
        
        logger.warning(f"Alert triggered: {rule.name} ({alert.severity})")
    
    async def _maybe_resolve_alert(self, rule_id: str) -> None:
        """Maybe resolve an alert if condition is no longer true."""
        # Find active alert for this rule
        alert_to_resolve = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id:
                alert_to_resolve = alert
                break
        
        if alert_to_resolve:
            alert_to_resolve.resolve()
            
            # Move to history
            self.alert_history.append(alert_to_resolve)
            del self.active_alerts[alert_to_resolve.alert_id]
            
            logger.info(f"Auto-resolved alert: {rule_id}")
    
    async def _send_alert_notification(self, alert: Alert) -> None:
        """Send alert notification to system queue."""
        try:
            await self.queue_manager.enqueue(
                'system.alerts',
                {
                    'type': 'alert',
                    'alert_id': alert.alert_id,
                    'rule_id': alert.rule_id,
                    'severity': alert.severity,
                    'message': alert.message,
                    'triggered_at': alert.triggered_at.isoformat(),
                    'metadata': alert.metadata
                },
                domain='system'
            )
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics and alerts."""
        # Clean up old metrics
        metrics_cutoff = datetime.utcnow() - timedelta(hours=self.config['metrics_retention_hours'])
        
        while self.metrics_history and self.metrics_history[0]['timestamp'] < metrics_cutoff:
            self.metrics_history.popleft()
        
        # Clean up old alerts
        alert_cutoff = datetime.utcnow() - timedelta(days=self.config['alert_retention_days'])
        
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.triggered_at >= alert_cutoff
        ]
        
        logger.debug("Cleaned up old monitoring data")
    
    async def get_monitor_status(self) -> Dict[str, Any]:
        """Get monitor status and statistics."""
        return {
            'running': self.running,
            'metrics_count': len(self.metrics_history),
            'active_alerts': len(self.active_alerts),
            'alert_rules': len(self.alert_rules),
            'alert_history_count': len(self.alert_history),
            'config': self.config.copy(),
            'current_metrics': self.aggregated_metrics.copy() if self.aggregated_metrics else None
        }
