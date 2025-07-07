"""
LLMFlow Optimization Molecules

This module contains molecules for system optimization and performance improvement.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import logging
import statistics
from dataclasses import dataclass, field

from ..core.base import DataAtom, ServiceAtom, ValidationResult
from ..atoms.data import StringAtom, BooleanAtom, IntegerAtom
from ..queue import QueueManager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    latency_ms: float
    throughput_ops_per_sec: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    queue_depth: int
    processed_messages: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'latency_ms': self.latency_ms,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'error_rate': self.error_rate,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'queue_depth': self.queue_depth,
            'processed_messages': self.processed_messages
        }


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation data structure."""
    recommendation_id: str
    target_component: str
    recommendation_type: str
    description: str
    expected_improvement: float
    confidence_score: float
    implementation_code: Optional[str] = None
    rollback_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'recommendation_id': self.recommendation_id,
            'target_component': self.target_component,
            'recommendation_type': self.recommendation_type,
            'description': self.description,
            'expected_improvement': self.expected_improvement,
            'confidence_score': self.confidence_score,
            'implementation_code': self.implementation_code,
            'rollback_code': self.rollback_code,
            'metadata': self.metadata
        }


class PerformanceMetricsAtom(DataAtom):
    """Data atom for performance metrics."""
    
    def __init__(self, metrics: PerformanceMetrics, metadata: Dict[str, Any] = None):
        super().__init__(metrics.to_dict(), metadata)
        self.metrics = metrics
    
    def validate(self) -> ValidationResult:
        """Validate performance metrics."""
        if self.metrics.latency_ms < 0:
            return ValidationResult.error("Latency cannot be negative")
        
        if self.metrics.throughput_ops_per_sec < 0:
            return ValidationResult.error("Throughput cannot be negative")
        
        if not (0 <= self.metrics.error_rate <= 1):
            return ValidationResult.error("Error rate must be between 0 and 1")
        
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize metrics to bytes."""
        import msgpack
        return msgpack.packb(self.value)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'PerformanceMetricsAtom':
        """Deserialize bytes to metrics."""
        import msgpack
        value = msgpack.unpackb(data, raw=False)
        
        metrics = PerformanceMetrics(
            timestamp=datetime.fromisoformat(value['timestamp']),
            latency_ms=value['latency_ms'],
            throughput_ops_per_sec=value['throughput_ops_per_sec'],
            error_rate=value['error_rate'],
            memory_usage_mb=value['memory_usage_mb'],
            cpu_usage_percent=value['cpu_usage_percent'],
            queue_depth=value['queue_depth'],
            processed_messages=value['processed_messages']
        )
        
        return cls(metrics)


class OptimizationRecommendationAtom(DataAtom):
    """Data atom for optimization recommendations."""
    
    def __init__(self, recommendation: OptimizationRecommendation, metadata: Dict[str, Any] = None):
        super().__init__(recommendation.to_dict(), metadata)
        self.recommendation = recommendation
    
    def validate(self) -> ValidationResult:
        """Validate optimization recommendation."""
        if not self.recommendation.recommendation_id:
            return ValidationResult.error("Recommendation ID is required")
        
        if not self.recommendation.target_component:
            return ValidationResult.error("Target component is required")
        
        if not (0 <= self.recommendation.confidence_score <= 1):
            return ValidationResult.error("Confidence score must be between 0 and 1")
        
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize recommendation to bytes."""
        import msgpack
        return msgpack.packb(self.value)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'OptimizationRecommendationAtom':
        """Deserialize bytes to recommendation."""
        import msgpack
        value = msgpack.unpackb(data, raw=False)
        
        recommendation = OptimizationRecommendation(
            recommendation_id=value['recommendation_id'],
            target_component=value['target_component'],
            recommendation_type=value['recommendation_type'],
            description=value['description'],
            expected_improvement=value['expected_improvement'],
            confidence_score=value['confidence_score'],
            implementation_code=value.get('implementation_code'),
            rollback_code=value.get('rollback_code'),
            metadata=value.get('metadata', {})
        )
        
        return cls(recommendation)


class PerformanceAnalysisMolecule(ServiceAtom):
    """Service molecule for performance analysis."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="performance_analysis_molecule",
            input_types=[
                "llmflow.molecules.optimization.PerformanceMetricsAtom"
            ],
            output_types=[
                "llmflow.atoms.data.StringAtom",  # Analysis report
                "llmflow.atoms.data.BooleanAtom"  # Performance issues detected
            ]
        )
        self.queue_manager = queue_manager
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        
        # Performance thresholds
        self.thresholds = {
            'max_latency_ms': 1000,
            'min_throughput_ops_per_sec': 100,
            'max_error_rate': 0.05,
            'max_memory_usage_mb': 1024,
            'max_cpu_usage_percent': 80,
            'max_queue_depth': 1000
        }
    
    async def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Process performance analysis."""
        metrics_atom = inputs[0]
        
        if not isinstance(metrics_atom, PerformanceMetricsAtom):
            return [StringAtom("Invalid metrics data"), BooleanAtom(False)]
        
        # Validate metrics
        validation_result = metrics_atom.validate()
        if not validation_result.is_valid:
            return [StringAtom(f"Invalid metrics: {validation_result.errors}"), BooleanAtom(False)]
        
        # Add to history
        self.metrics_history.append(metrics_atom.metrics)
        
        # Keep history size manageable
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        # Analyze performance
        analysis_report = await self._analyze_performance(metrics_atom.metrics)
        issues_detected = await self._detect_performance_issues(metrics_atom.metrics)
        
        logger.info(f"Performance analysis completed: issues_detected={issues_detected}")
        return [StringAtom(analysis_report), BooleanAtom(issues_detected)]
    
    async def _analyze_performance(self, metrics: PerformanceMetrics) -> str:
        """Analyze performance metrics."""
        analysis = {
            'timestamp': metrics.timestamp.isoformat(),
            'current_metrics': metrics.to_dict(),
            'threshold_violations': [],
            'trends': {},
            'recommendations': []
        }
        
        # Check threshold violations
        if metrics.latency_ms > self.thresholds['max_latency_ms']:
            analysis['threshold_violations'].append(
                f"High latency: {metrics.latency_ms}ms > {self.thresholds['max_latency_ms']}ms"
            )
        
        if metrics.throughput_ops_per_sec < self.thresholds['min_throughput_ops_per_sec']:
            analysis['threshold_violations'].append(
                f"Low throughput: {metrics.throughput_ops_per_sec} < {self.thresholds['min_throughput_ops_per_sec']} ops/sec"
            )
        
        if metrics.error_rate > self.thresholds['max_error_rate']:
            analysis['threshold_violations'].append(
                f"High error rate: {metrics.error_rate} > {self.thresholds['max_error_rate']}"
            )
        
        if metrics.memory_usage_mb > self.thresholds['max_memory_usage_mb']:
            analysis['threshold_violations'].append(
                f"High memory usage: {metrics.memory_usage_mb}MB > {self.thresholds['max_memory_usage_mb']}MB"
            )
        
        if metrics.cpu_usage_percent > self.thresholds['max_cpu_usage_percent']:
            analysis['threshold_violations'].append(
                f"High CPU usage: {metrics.cpu_usage_percent}% > {self.thresholds['max_cpu_usage_percent']}%"
            )
        
        if metrics.queue_depth > self.thresholds['max_queue_depth']:
            analysis['threshold_violations'].append(
                f"High queue depth: {metrics.queue_depth} > {self.thresholds['max_queue_depth']}"
            )
        
        # Generate recommendations
        analysis['recommendations'] = await self._generate_recommendations(metrics)
        
        return json.dumps(analysis, indent=2)
    
    async def _detect_performance_issues(self, metrics: PerformanceMetrics) -> bool:
        """Detect if there are performance issues."""
        issues = []
        
        # Check each threshold
        if metrics.latency_ms > self.thresholds['max_latency_ms']:
            issues.append("high_latency")
        
        if metrics.throughput_ops_per_sec < self.thresholds['min_throughput_ops_per_sec']:
            issues.append("low_throughput")
        
        if metrics.error_rate > self.thresholds['max_error_rate']:
            issues.append("high_error_rate")
        
        if metrics.memory_usage_mb > self.thresholds['max_memory_usage_mb']:
            issues.append("high_memory_usage")
        
        if metrics.cpu_usage_percent > self.thresholds['max_cpu_usage_percent']:
            issues.append("high_cpu_usage")
        
        if metrics.queue_depth > self.thresholds['max_queue_depth']:
            issues.append("high_queue_depth")
        
        return len(issues) > 0
    
    async def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if metrics.latency_ms > self.thresholds['max_latency_ms']:
            recommendations.append("Consider optimizing queue processing logic")
            recommendations.append("Implement caching for frequently accessed data")
        
        if metrics.throughput_ops_per_sec < self.thresholds['min_throughput_ops_per_sec']:
            recommendations.append("Consider increasing worker pool size")
            recommendations.append("Optimize message serialization/deserialization")
        
        if metrics.error_rate > self.thresholds['max_error_rate']:
            recommendations.append("Review error handling and retry logic")
            recommendations.append("Implement circuit breakers for external dependencies")
        
        if metrics.memory_usage_mb > self.thresholds['max_memory_usage_mb']:
            recommendations.append("Implement memory pooling for frequent allocations")
            recommendations.append("Review and optimize data structures")
        
        if metrics.cpu_usage_percent > self.thresholds['max_cpu_usage_percent']:
            recommendations.append("Consider horizontal scaling")
            recommendations.append("Optimize CPU-intensive operations")
        
        if metrics.queue_depth > self.thresholds['max_queue_depth']:
            recommendations.append("Increase processing capacity")
            recommendations.append("Implement queue prioritization")
        
        return recommendations


class OptimizationRecommendationMolecule(ServiceAtom):
    """Service molecule for generating optimization recommendations."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="optimization_recommendation_molecule",
            input_types=[
                "llmflow.atoms.data.StringAtom",  # Performance analysis report
                "llmflow.atoms.data.StringAtom"   # Component code
            ],
            output_types=[
                "llmflow.molecules.optimization.OptimizationRecommendationAtom"
            ]
        )
        self.queue_manager = queue_manager
        self.recommendation_templates = {
            'latency_optimization': {
                'description': 'Optimize component for reduced latency',
                'expected_improvement': 0.3,
                'confidence_score': 0.8
            },
            'throughput_optimization': {
                'description': 'Optimize component for increased throughput',
                'expected_improvement': 0.5,
                'confidence_score': 0.7
            },
            'memory_optimization': {
                'description': 'Optimize component for reduced memory usage',
                'expected_improvement': 0.4,
                'confidence_score': 0.6
            },
            'error_reduction': {
                'description': 'Improve error handling and reliability',
                'expected_improvement': 0.8,
                'confidence_score': 0.9
            }
        }
    
    async def process(self, inputs: List[DataAtom]) -> List[OptimizationRecommendationAtom]:
        """Process optimization recommendation generation."""
        analysis_report = inputs[0]
        component_code = inputs[1]
        
        if not isinstance(analysis_report, StringAtom) or not isinstance(component_code, StringAtom):
            return [OptimizationRecommendationAtom(
                OptimizationRecommendation(
                    recommendation_id="error",
                    target_component="unknown",
                    recommendation_type="error",
                    description="Invalid input data",
                    expected_improvement=0.0,
                    confidence_score=0.0
                )
            )]
        
        # Parse analysis report
        try:
            analysis = json.loads(analysis_report.value)
        except json.JSONDecodeError:
            return [OptimizationRecommendationAtom(
                OptimizationRecommendation(
                    recommendation_id="error",
                    target_component="unknown",
                    recommendation_type="error",
                    description="Invalid analysis report format",
                    expected_improvement=0.0,
                    confidence_score=0.0
                )
            )]
        
        # Generate recommendation based on analysis
        recommendation = await self._generate_optimization_recommendation(analysis, component_code.value)
        
        logger.info(f"Generated optimization recommendation: {recommendation.recommendation_id}")
        return [OptimizationRecommendationAtom(recommendation)]
    
    async def _generate_optimization_recommendation(self, analysis: Dict[str, Any], 
                                                  component_code: str) -> OptimizationRecommendation:
        """Generate optimization recommendation based on analysis."""
        import uuid
        
        # Determine primary issue
        threshold_violations = analysis.get('threshold_violations', [])
        
        if not threshold_violations:
            return OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                target_component="system",
                recommendation_type="no_optimization_needed",
                description="No performance issues detected",
                expected_improvement=0.0,
                confidence_score=1.0
            )
        
        # Identify the most critical issue
        primary_issue = await self._identify_primary_issue(threshold_violations)
        
        # Get recommendation template
        template = self.recommendation_templates.get(primary_issue, {
            'description': 'General performance optimization',
            'expected_improvement': 0.2,
            'confidence_score': 0.5
        })
        
        # Generate implementation code
        implementation_code = await self._generate_implementation_code(primary_issue, component_code)
        rollback_code = await self._generate_rollback_code(component_code)
        
        recommendation = OptimizationRecommendation(
            recommendation_id=str(uuid.uuid4()),
            target_component=self._extract_component_name(component_code),
            recommendation_type=primary_issue,
            description=template['description'],
            expected_improvement=template['expected_improvement'],
            confidence_score=template['confidence_score'],
            implementation_code=implementation_code,
            rollback_code=rollback_code,
            metadata={
                'analysis_timestamp': analysis.get('timestamp'),
                'violations': threshold_violations
            }
        )
        
        return recommendation
    
    async def _identify_primary_issue(self, violations: List[str]) -> str:
        """Identify the primary performance issue."""
        detected_issues = []
        
        for violation in violations:
            if 'error rate' in violation.lower():
                detected_issues.append(('error_reduction', 1))
            elif 'latency' in violation.lower():
                detected_issues.append(('latency_optimization', 2))
            elif 'memory' in violation.lower():
                detected_issues.append(('memory_optimization', 3))
            elif 'throughput' in violation.lower():
                detected_issues.append(('throughput_optimization', 4))
        
        if detected_issues:
            detected_issues.sort(key=lambda x: x[1])
            return detected_issues[0][0]
        
        return 'latency_optimization'
    
    async def _generate_implementation_code(self, issue_type: str, original_code: str) -> str:
        """Generate implementation code for the optimization."""
        if issue_type == 'latency_optimization':
            return """# Latency optimization: Add caching and async processing
import asyncio
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_operation(key):
    pass

async def optimized_batch_process(items):
    tasks = [process_item_async(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results"""
        
        elif issue_type == 'throughput_optimization':
            return """# Throughput optimization: Use connection pooling and batch processing
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def optimized_process(items):
    batch_size = 100
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        await process_batch_async(batch)"""
        
        return "# No specific optimization code generated"
    
    async def _generate_rollback_code(self, original_code: str) -> str:
        """Generate rollback code."""
        return f"# Rollback to original implementation\n{original_code}"
    
    def _extract_component_name(self, component_code: str) -> str:
        """Extract component name from code."""
        lines = component_code.split('\n')
        for line in lines:
            if line.strip().startswith('class '):
                return line.strip().split()[1].split('(')[0]
            elif line.strip().startswith('def '):
                return line.strip().split()[1].split('(')[0]
        
        return "unknown_component"
