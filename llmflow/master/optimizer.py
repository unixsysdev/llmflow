"""
LLMFlow Master Queue Optimizer

This module implements the LLM-based optimization system for LLMFlow,
analyzing performance and generating code improvements.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import uuid

from ..queue import QueueManager
from ..molecules.optimization import (
    PerformanceMetrics, OptimizationRecommendation,
    PerformanceMetricsAtom, OptimizationRecommendationAtom,
    PerformanceAnalysisMolecule, OptimizationRecommendationMolecule
)
from ..atoms.data import StringAtom

logger = logging.getLogger(__name__)


@dataclass
class OptimizationContext:
    """Context for optimization analysis."""
    component_id: str
    component_name: str
    component_type: str
    source_code: str
    metrics_history: List[PerformanceMetrics] = field(default_factory=list)
    optimization_history: List[OptimizationRecommendation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationTask:
    """A task for optimization analysis."""
    task_id: str
    context: OptimizationContext
    priority: int = 5  # 1-10, higher is more priority
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[OptimizationRecommendation] = None
    error: Optional[str] = None


class LLMOptimizer:
    """LLM-based code optimization system."""
    
    def __init__(self, queue_manager: QueueManager):
        self.queue_manager = queue_manager
        self.running = False
        
        # Analysis molecules
        self.performance_analysis = PerformanceAnalysisMolecule(queue_manager)
        self.optimization_recommendation = OptimizationRecommendationMolecule(queue_manager)
        
        # Task management
        self.optimization_tasks: Dict[str, OptimizationTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
        # Component tracking
        self.component_contexts: Dict[str, OptimizationContext] = {}
        
        # Configuration
        self.config = {
            'max_concurrent_tasks': 5,
            'task_timeout_seconds': 300,
            'metrics_threshold_for_optimization': 10,
            'optimization_cooldown_minutes': 60,
            'max_optimization_attempts': 3
        }
        
        # Worker tasks
        self.worker_tasks: List[asyncio.Task] = []
        
        # Metrics
        self.optimizer_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'active_tasks': 0,
            'recommendations_generated': 0,
            'optimizations_applied': 0
        }
    
    async def start(self) -> None:
        """Start the LLM optimizer."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker tasks
        for i in range(self.config['max_concurrent_tasks']):
            worker_task = asyncio.create_task(self._optimization_worker(f"worker_{i}"))
            self.worker_tasks.append(worker_task)
        
        logger.info(f"LLM optimizer started with {len(self.worker_tasks)} workers")
    
    async def stop(self) -> None:
        """Stop the LLM optimizer."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        
        logger.info("LLM optimizer stopped")
    
    async def analyze_component(self, component_id: str, component_name: str, 
                               component_type: str, source_code: str,
                               metrics: List[PerformanceMetrics]) -> str:
        """Analyze a component for optimization opportunities."""
        # Create or update context
        if component_id not in self.component_contexts:
            self.component_contexts[component_id] = OptimizationContext(
                component_id=component_id,
                component_name=component_name,
                component_type=component_type,
                source_code=source_code
            )
        
        context = self.component_contexts[component_id]
        
        # Update metrics history
        context.metrics_history.extend(metrics)
        
        # Keep only recent metrics
        if len(context.metrics_history) > 100:
            context.metrics_history = context.metrics_history[-100:]
        
        # Check if optimization is needed
        if await self._should_optimize(context):
            # Create optimization task
            task = OptimizationTask(
                task_id=str(uuid.uuid4()),
                context=context,
                priority=await self._calculate_priority(context)
            )
            
            # Queue task
            await self.task_queue.put(task)
            self.optimization_tasks[task.task_id] = task
            
            self.optimizer_metrics['total_tasks'] += 1
            self.optimizer_metrics['active_tasks'] += 1
            
            logger.info(f"Queued optimization task for component {component_name}")
            return task.task_id
        
        return "no_optimization_needed"
    
    async def _should_optimize(self, context: OptimizationContext) -> bool:
        """Determine if a component should be optimized."""
        # Check if we have enough metrics
        if len(context.metrics_history) < self.config['metrics_threshold_for_optimization']:
            return False
        
        # Check cooldown period
        if context.optimization_history:
            last_optimization = context.optimization_history[-1]
            time_since_last = datetime.utcnow() - datetime.fromisoformat(last_optimization.metadata.get('created_at', '2000-01-01'))
            cooldown_period = self.config['optimization_cooldown_minutes'] * 60
            
            if time_since_last.total_seconds() < cooldown_period:
                return False
        
        # Check if performance is degrading
        recent_metrics = context.metrics_history[-10:]
        if len(recent_metrics) < 5:
            return False
        
        # Simple heuristic: check if recent performance is worse than historical average
        if len(context.metrics_history) > 10:
            historical_avg_latency = sum(m.latency_ms for m in context.metrics_history[:-10]) / len(context.metrics_history[:-10])
            recent_avg_latency = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
            
            # If recent latency is 20% worse than historical average, optimize
            if recent_avg_latency > historical_avg_latency * 1.2:
                return True
        
        # Check error rate
        recent_avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        if recent_avg_error_rate > 0.05:  # 5% error rate threshold
            return True
        
        # Check memory usage
        recent_avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        if recent_avg_memory > 1024:  # 1GB threshold
            return True
        
        return False
    
    async def _calculate_priority(self, context: OptimizationContext) -> int:
        """Calculate optimization priority based on context."""
        priority = 5  # Default priority
        
        if not context.metrics_history:
            return priority
        
        recent_metrics = context.metrics_history[-5:]
        
        # Higher priority for high error rates
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        if avg_error_rate > 0.1:
            priority += 3
        elif avg_error_rate > 0.05:
            priority += 2
        
        # Higher priority for high latency
        avg_latency = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
        if avg_latency > 2000:
            priority += 2
        elif avg_latency > 1000:
            priority += 1
        
        # Higher priority for critical components
        if context.component_type in ['master', 'conductor']:
            priority += 2
        
        return min(priority, 10)  # Cap at 10
    
    async def _optimization_worker(self, worker_id: str) -> None:
        """Worker task for processing optimization requests."""
        logger.info(f"Optimization worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Process the task
                await self._process_optimization_task(task, worker_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in optimization worker {worker_id}: {e}")
        
        logger.info(f"Optimization worker {worker_id} stopped")
    
    async def _process_optimization_task(self, task: OptimizationTask, worker_id: str) -> None:
        """Process an optimization task."""
        task.status = "processing"
        
        try:
            logger.info(f"Worker {worker_id} processing optimization task {task.task_id}")
            
            # Analyze performance
            analysis_result = await self._analyze_performance(task.context)
            
            # Generate optimization recommendation
            recommendation = await self._generate_optimization_recommendation(
                task.context, analysis_result
            )
            
            # Store result
            task.result = recommendation
            task.status = "completed"
            
            # Update context
            task.context.optimization_history.append(recommendation)
            
            # Update metrics
            self.optimizer_metrics['completed_tasks'] += 1
            self.optimizer_metrics['active_tasks'] -= 1
            self.optimizer_metrics['recommendations_generated'] += 1
            
            # Send result to optimization queue
            await self._send_optimization_result(task)
            
            logger.info(f"Completed optimization task {task.task_id}")
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            
            self.optimizer_metrics['failed_tasks'] += 1
            self.optimizer_metrics['active_tasks'] -= 1
            
            logger.error(f"Failed to process optimization task {task.task_id}: {e}")
    
    async def _analyze_performance(self, context: OptimizationContext) -> str:
        """Analyze performance metrics."""
        if not context.metrics_history:
            return json.dumps({'error': 'No metrics available'})
        
        # Get recent metrics
        recent_metrics = context.metrics_history[-1]
        metrics_atom = PerformanceMetricsAtom(recent_metrics)
        
        # Analyze using performance analysis molecule
        analysis_result = await self.performance_analysis.process([metrics_atom])
        
        if analysis_result and len(analysis_result) > 0:
            return analysis_result[0].value
        
        return json.dumps({'error': 'Analysis failed'})
    
    async def _generate_optimization_recommendation(self, context: OptimizationContext, 
                                                   analysis_report: str) -> OptimizationRecommendation:
        """Generate optimization recommendation."""
        # Prepare input atoms
        analysis_atom = StringAtom(analysis_report)
        code_atom = StringAtom(context.source_code)
        
        # Generate recommendation using optimization molecule
        recommendation_result = await self.optimization_recommendation.process([
            analysis_atom, code_atom
        ])
        
        if recommendation_result and len(recommendation_result) > 0:
            return recommendation_result[0].recommendation
        
        # Fallback recommendation
        return OptimizationRecommendation(
            recommendation_id=str(uuid.uuid4()),
            target_component=context.component_name,
            recommendation_type="general_optimization",
            description="General performance optimization needed",
            expected_improvement=0.1,
            confidence_score=0.5,
            metadata={
                'created_at': datetime.utcnow().isoformat(),
                'fallback': True
            }
        )
    
    async def _send_optimization_result(self, task: OptimizationTask) -> None:
        """Send optimization result to the optimization queue."""
        try:
            result_data = {
                'task_id': task.task_id,
                'component_id': task.context.component_id,
                'component_name': task.context.component_name,
                'status': task.status,
                'recommendation': task.result.to_dict() if task.result else None,
                'error': task.error,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self.queue_manager.enqueue(
                'system.optimization',
                result_data,
                domain='system'
            )
            
        except Exception as e:
            logger.error(f"Error sending optimization result: {e}")
    
    async def get_optimizer_metrics(self) -> Dict[str, Any]:
        """Get optimizer metrics."""
        return {
            'metrics': self.optimizer_metrics.copy(),
            'config': self.config.copy(),
            'active_components': len(self.component_contexts),
            'queued_tasks': self.task_queue.qsize(),
            'total_tasks': len(self.optimization_tasks),
            'worker_count': len(self.worker_tasks),
            'timestamp': datetime.utcnow().isoformat()
        }
