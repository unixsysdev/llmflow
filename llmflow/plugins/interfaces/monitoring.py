"""
Monitoring Provider Interface

This module defines the interface for monitoring providers in the LLMFlow framework.
Monitoring providers handle metrics collection, alerting, and observability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, AsyncIterator
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringError(Exception):
    """Base exception for monitoring-related errors."""
    pass


class MetricError(MonitoringError):
    """Raised when metric operations fail."""
    pass


class AlertError(MonitoringError):
    """Raised when alert operations fail."""
    pass


class Metric:
    """
    Represents a metric data point.
    """
    
    def __init__(self, 
                 name: str, 
                 value: float,
                 metric_type: MetricType,
                 tags: Optional[Dict[str, str]] = None,
                 timestamp: Optional[datetime] = None):
        self.name = name
        self.value = value
        self.metric_type = metric_type
        self.tags = tags or {}
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'tags': self.tags,
            'timestamp': self.timestamp.isoformat()
        }


class Alert:
    """
    Represents an alert.
    """
    
    def __init__(self, 
                 name: str, 
                 message: str,
                 severity: AlertSeverity,
                 tags: Optional[Dict[str, str]] = None,
                 timestamp: Optional[datetime] = None,
                 resolved: bool = False):
        self.name = name
        self.message = message
        self.severity = severity
        self.tags = tags or {}
        self.timestamp = timestamp or datetime.utcnow()
        self.resolved = resolved
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'name': self.name,
            'message': self.message,
            'severity': self.severity.value,
            'tags': self.tags,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved
        }


class Timer:
    """
    Timer for measuring durations.
    """
    
    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.tags = tags or {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = datetime.utcnow()
    
    def stop(self) -> float:
        """
        Stop the timer and return duration.
        
        Returns:
            Duration in seconds
        """
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = datetime.utcnow()
        duration = (self.end_time - self.start_time).total_seconds()
        return duration
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class IMonitoringProvider(ABC):
    """
    Interface for monitoring providers in LLMFlow.
    
    This interface defines the contract for all monitoring implementations,
    including Prometheus, StatsD, CloudWatch, and other monitoring systems.
    """
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the monitoring provider name.
        
        Returns:
            The name of the monitoring provider
        """
        pass
    
    @abstractmethod
    def get_supported_metric_types(self) -> List[MetricType]:
        """
        Get the list of supported metric types.
        
        Returns:
            List of supported metric types
        """
        pass
    
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> None:
        """
        Connect to the monitoring system.
        
        Args:
            config: Monitoring provider configuration
            
        Raises:
            MonitoringError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the monitoring system.
        
        Raises:
            MonitoringError: If disconnection fails
        """
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if connected to the monitoring system.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    async def record_metric(self, metric: Metric) -> None:
        """
        Record a metric.
        
        Args:
            metric: Metric to record
            
        Raises:
            MetricError: If metric recording fails
        """
        pass
    
    @abstractmethod
    async def record_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a counter metric.
        
        Args:
            name: Metric name
            value: Counter value to add
            tags: Optional tags
            
        Raises:
            MetricError: If metric recording fails
        """
        pass
    
    @abstractmethod
    async def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a gauge metric.
        
        Args:
            name: Metric name
            value: Gauge value
            tags: Optional tags
            
        Raises:
            MetricError: If metric recording fails
        """
        pass
    
    @abstractmethod
    async def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a histogram metric.
        
        Args:
            name: Metric name
            value: Histogram value
            tags: Optional tags
            
        Raises:
            MetricError: If metric recording fails
        """
        pass
    
    @abstractmethod
    async def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a timer metric.
        
        Args:
            name: Metric name
            duration: Duration in seconds
            tags: Optional tags
            
        Raises:
            MetricError: If metric recording fails
        """
        pass
    
    @abstractmethod
    def create_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> Timer:
        """
        Create a timer for measuring durations.
        
        Args:
            name: Timer name
            tags: Optional tags
            
        Returns:
            Timer instance
        """
        pass
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> None:
        """
        Send an alert.
        
        Args:
            alert: Alert to send
            
        Raises:
            AlertError: If alert sending fails
        """
        pass
    
    @abstractmethod
    async def resolve_alert(self, alert_name: str) -> None:
        """
        Resolve an alert.
        
        Args:
            alert_name: Name of the alert to resolve
            
        Raises:
            AlertError: If alert resolution fails
        """
        pass
    
    @abstractmethod
    async def get_active_alerts(self) -> List[Alert]:
        """
        Get all active alerts.
        
        Returns:
            List of active alerts
            
        Raises:
            AlertError: If alert retrieval fails
        """
        pass
    
    @abstractmethod
    async def query_metrics(self, 
                           query: str, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> AsyncIterator[Metric]:
        """
        Query metrics using provider-specific query language.
        
        Args:
            query: Query string
            start_time: Optional start time
            end_time: Optional end time
            
        Yields:
            Metric data points
            
        Raises:
            MetricError: If query fails
        """
        pass
    
    @abstractmethod
    async def get_metric_names(self) -> List[str]:
        """
        Get all available metric names.
        
        Returns:
            List of metric names
            
        Raises:
            MetricError: If metric name retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_metric_tags(self, metric_name: str) -> List[str]:
        """
        Get available tags for a metric.
        
        Args:
            metric_name: Metric name
            
        Returns:
            List of tag names
            
        Raises:
            MetricError: If tag retrieval fails
        """
        pass
    
    @abstractmethod
    async def create_dashboard(self, name: str, config: Dict[str, Any]) -> str:
        """
        Create a monitoring dashboard.
        
        Args:
            name: Dashboard name
            config: Dashboard configuration
            
        Returns:
            Dashboard ID
            
        Raises:
            MonitoringError: If dashboard creation fails
        """
        pass
    
    @abstractmethod
    async def update_dashboard(self, dashboard_id: str, config: Dict[str, Any]) -> None:
        """
        Update a monitoring dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            config: Updated dashboard configuration
            
        Raises:
            MonitoringError: If dashboard update fails
        """
        pass
    
    @abstractmethod
    async def delete_dashboard(self, dashboard_id: str) -> None:
        """
        Delete a monitoring dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            
        Raises:
            MonitoringError: If dashboard deletion fails
        """
        pass
    
    @abstractmethod
    async def get_dashboards(self) -> List[Dict[str, Any]]:
        """
        Get all dashboards.
        
        Returns:
            List of dashboard information
            
        Raises:
            MonitoringError: If dashboard retrieval fails
        """
        pass
    
    @abstractmethod
    async def create_alert_rule(self, name: str, condition: str, severity: AlertSeverity) -> str:
        """
        Create an alert rule.
        
        Args:
            name: Alert rule name
            condition: Alert condition
            severity: Alert severity
            
        Returns:
            Alert rule ID
            
        Raises:
            AlertError: If alert rule creation fails
        """
        pass
    
    @abstractmethod
    async def update_alert_rule(self, rule_id: str, condition: str, severity: AlertSeverity) -> None:
        """
        Update an alert rule.
        
        Args:
            rule_id: Alert rule ID
            condition: Updated alert condition
            severity: Updated alert severity
            
        Raises:
            AlertError: If alert rule update fails
        """
        pass
    
    @abstractmethod
    async def delete_alert_rule(self, rule_id: str) -> None:
        """
        Delete an alert rule.
        
        Args:
            rule_id: Alert rule ID
            
        Raises:
            AlertError: If alert rule deletion fails
        """
        pass
    
    @abstractmethod
    async def get_alert_rules(self) -> List[Dict[str, Any]]:
        """
        Get all alert rules.
        
        Returns:
            List of alert rule information
            
        Raises:
            AlertError: If alert rule retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get monitoring provider statistics.
        
        Returns:
            Dictionary containing monitoring statistics
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check the health of the monitoring system.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def batch_record_metrics(self, metrics: List[Metric]) -> None:
        """
        Record multiple metrics in a batch.
        
        Args:
            metrics: List of metrics to record
            
        Raises:
            MetricError: If batch recording fails
        """
        pass
    
    @abstractmethod
    async def set_global_tags(self, tags: Dict[str, str]) -> None:
        """
        Set global tags that will be added to all metrics.
        
        Args:
            tags: Global tags to set
            
        Raises:
            MetricError: If setting global tags fails
        """
        pass
    
    @abstractmethod
    async def get_global_tags(self) -> Dict[str, str]:
        """
        Get current global tags.
        
        Returns:
            Current global tags
        """
        pass
