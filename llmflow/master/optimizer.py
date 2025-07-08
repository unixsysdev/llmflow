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
from ..molecules.llm_optimization import (
    LLMCodeAnalysisMolecule, LLMOptimizationGeneratorMolecule, LLMSystemOptimizationMolecule
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
        
        # NEW: LLM-powered analysis molecules
        self.llm_code_analysis = LLMCodeAnalysisMolecule(queue_manager)
        self.llm_optimization_generator = LLMOptimizationGeneratorMolecule(queue_manager)
        self.llm_system_optimization = LLMSystemOptimizationMolecule(queue_manager)
        
        # Legacy molecules for fallback
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
            'max_optimization_attempts': 3,
            'use_llm_optimization': True,  # NEW: Enable LLM optimization
            'llm_fallback_enabled': True   # NEW: Fallback to simple optimization
        }
        
        # Worker tasks
        self.worker_tasks: List[asyncio.Task] = []
        
        # Enhanced metrics
        self.optimizer_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'active_tasks': 0,
            'recommendations_generated': 0,
            'optimizations_applied': 0,
            'llm_analyses_performed': 0,
            'llm_recommendations_generated': 0,
            'fallback_optimizations': 0
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
        """Analyze performance metrics using LLM-powered analysis."""
        if not context.metrics_history:
            return json.dumps({'error': 'No metrics available'})
        
        try:
            # Use LLM analysis if enabled and available
            if self.config.get('use_llm_optimization', True):
                # Prepare metrics data
                recent_metrics = context.metrics_history[-5:]  # Last 5 metrics
                metrics_json = json.dumps([m.to_dict() for m in recent_metrics], indent=2)
                
                # Use LLM code analysis molecule
                analysis_result = await self.llm_code_analysis.process([
                    StringAtom(context.source_code),
                    StringAtom(metrics_json)
                ])
                
                if analysis_result and len(analysis_result) >= 2:
                    analysis_content = analysis_result[0].value
                    issues_found = analysis_result[1].value
                    
                    self.optimizer_metrics['llm_analyses_performed'] += 1
                    
                    logger.info(f"LLM analysis completed for {context.component_name}, issues: {issues_found}")
                    return analysis_content
        
        except Exception as e:
            logger.warning(f"LLM analysis failed for {context.component_name}: {e}")
        
        # Fallback to traditional analysis
        if self.config.get('llm_fallback_enabled', True):
            logger.info(f"Using fallback analysis for {context.component_name}")
            
            recent_metrics = context.metrics_history[-1]
            metrics_atom = PerformanceMetricsAtom(recent_metrics)
            
            analysis_result = await self.performance_analysis.process([metrics_atom])
            
            if analysis_result and len(analysis_result) > 0:
                self.optimizer_metrics['fallback_optimizations'] += 1
                return analysis_result[0].value
        
        return json.dumps({'error': 'All analysis methods failed'})

    
    async def _generate_optimization_recommendation(self, context: OptimizationContext, 
                                                   analysis_report: str) -> OptimizationRecommendation:
        """Generate optimization recommendation using LLM."""
        try:
            # Determine optimization type from analysis
            optimization_type = await self._determine_optimization_type(analysis_report)
            
            # Use LLM optimization generator if enabled
            if self.config.get('use_llm_optimization', True):
                logger.info(f"Generating LLM optimization for {context.component_name}, type: {optimization_type}")
                
                recommendation_result = await self.llm_optimization_generator.process([
                    StringAtom(analysis_report),
                    StringAtom(context.source_code),
                    StringAtom(optimization_type)
                ])
                
                if recommendation_result and len(recommendation_result) > 0:
                    recommendation = recommendation_result[0].recommendation
                    
                    # Validate recommendation quality
                    if recommendation.confidence_score >= 0.5 and not recommendation.description.startswith("Optimization generation failed"):
                        self.optimizer_metrics['llm_recommendations_generated'] += 1
                        logger.info(f"LLM recommendation generated: {recommendation.recommendation_id}")
                        return recommendation
                    else:
                        logger.warning(f"Low quality LLM recommendation, falling back: {recommendation.description}")
        
        except Exception as e:
            logger.warning(f"LLM optimization generation failed for {context.component_name}: {e}")
        
        # Fallback to traditional optimization
        if self.config.get('llm_fallback_enabled', True):
            logger.info(f"Using fallback optimization for {context.component_name}")
            
            analysis_atom = StringAtom(analysis_report)
            code_atom = StringAtom(context.source_code)
            
            recommendation_result = await self.optimization_recommendation.process([
                analysis_atom, code_atom
            ])
            
            if recommendation_result and len(recommendation_result) > 0:
                self.optimizer_metrics['fallback_optimizations'] += 1
                return recommendation_result[0].recommendation
        
        # Ultimate fallback
        return OptimizationRecommendation(
            recommendation_id=str(uuid.uuid4()),
            target_component=context.component_name,
            optimization_type=optimization_type,
            description=f"Automated {optimization_type} optimization needed - analysis available but optimization generation failed",
            expected_improvement=0.1,
            confidence_score=0.3,
            metadata={
                'created_at': datetime.utcnow().isoformat(),
                'fallback': True,
                'analysis_available': True
            }
        )
    
    async def _determine_optimization_type(self, analysis_report: str) -> str:
        """Determine the primary optimization type from analysis report."""
        try:
            # Try to parse as JSON first
            analysis_data = json.loads(analysis_report)
            
            # Check for specific optimization opportunities
            if 'optimization_opportunities' in analysis_data:
                opportunities = analysis_data['optimization_opportunities']
                if opportunities:
                    # Return the highest priority optimization type
                    for opp in opportunities:
                        if opp.get('severity') in ['high', 'critical']:
                            return opp.get('type', 'latency_optimization')
                    
                    # Return first opportunity type if no high/critical
                    return opportunities[0].get('type', 'latency_optimization')
            
            # Check primary concerns
            if 'primary_concerns' in analysis_data:
                concerns = analysis_data['primary_concerns']
                if concerns:
                    concern_text = ' '.join(concerns).lower()
                    
                    if 'memory' in concern_text or 'leak' in concern_text:
                        return 'memory_optimization'
                    elif 'error' in concern_text or 'exception' in concern_text:
                        return 'error_reduction'
                    elif 'slow' in concern_text or 'latency' in concern_text:
                        return 'latency_optimization'
                    elif 'throughput' in concern_text or 'performance' in concern_text:
                        return 'throughput_optimization'
        
        except json.JSONDecodeError:
            # Fallback to text analysis
            analysis_lower = analysis_report.lower()
            
            if 'memory' in analysis_lower or 'leak' in analysis_lower:
                return 'memory_optimization'
            elif 'error' in analysis_lower or 'exception' in analysis_lower:
                return 'error_reduction'
            elif 'slow' in analysis_lower or 'latency' in analysis_lower:
                return 'latency_optimization'
            elif 'throughput' in analysis_lower:
                return 'throughput_optimization'
        
        # Default fallback
        return 'latency_optimization'
    
    async def analyze_system_performance(self) -> str:
        """Perform system-wide performance analysis using LLM."""
        try:
            # Collect system-wide data
            system_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_components': len(self.component_contexts),
                'active_optimizations': self.optimizer_metrics['active_tasks'],
                'component_summary': []
            }
            
            # Add component summaries
            for context in self.component_contexts.values():
                if context.metrics_history:
                    recent_metrics = context.metrics_history[-3:]  # Last 3 metrics
                    avg_latency = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
                    avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
                    
                    system_data['component_summary'].append({
                        'component_id': context.component_id,
                        'component_name': context.component_name,
                        'component_type': context.component_type,
                        'avg_latency_ms': avg_latency,
                        'avg_error_rate': avg_error_rate,
                        'optimization_count': len(context.optimization_history),
                        'last_optimization': context.optimization_history[-1].metadata.get('created_at') if context.optimization_history else None
                    })
            
            # Use LLM system optimization molecule
            if self.config.get('use_llm_optimization', True):
                system_data_json = json.dumps(system_data, indent=2)
                
                analysis_result = await self.llm_system_optimization.process([
                    StringAtom(system_data_json)
                ])
                
                if analysis_result and len(analysis_result) > 0:
                    logger.info("System-wide LLM analysis completed")
                    return analysis_result[0].value
            
            # Fallback to basic system analysis
            return json.dumps({
                'system_health': 'unknown',
                'message': 'Basic system data collected',
                'data': system_data
            }, indent=2)
        
        except Exception as e:
            logger.error(f"System performance analysis failed: {e}")
            return json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })    
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
    
        async def _identify_optimization_pattern(self, recent_metrics: List[PerformanceMetrics], 
                                               anomalies: List[str] = None) -> str:
            """Identify the optimization pattern based on metrics and anomalies."""
            if anomalies:
                # Map anomalies to optimization patterns
                if 'memory_leak' in anomalies:
                    return 'memory_optimization'
                elif 'latency_spike' in anomalies:
                    return 'latency_optimization'
                elif 'high_error_rate' in anomalies:
                    return 'error_reduction'
                elif 'cpu_spike' in anomalies:
                    return 'throughput_optimization'
            
            if not recent_metrics:
                return 'latency_optimization'  # Default
            
            # Analyze metrics to determine primary issue
            avg_latency = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
            avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput_ops_per_sec for m in recent_metrics) / len(recent_metrics)
            
            # Priority order: errors > memory > latency > throughput
            if avg_error_rate > 0.02:
                return 'error_reduction'
            elif len(recent_metrics) > 5:
                memory_trend = self._calculate_trend([m.memory_usage_mb for m in recent_metrics])
                if memory_trend > 1.0:  # Memory growing
                    return 'memory_optimization'
            
            if avg_latency > 100:  # High latency
                return 'latency_optimization'
            elif avg_throughput < 500:  # Low throughput
                return 'throughput_optimization'
            
            return 'latency_optimization'  # Default
        
        def _calculate_trend(self, values: List[float]) -> float:
            """Calculate trend (slope) of values over time."""
            if len(values) < 2:
                return 0
            
            n = len(values)
            x_sum = sum(range(n))
            y_sum = sum(values)
            xy_sum = sum(i * values[i] for i in range(n))
            x2_sum = sum(i * i for i in range(n))
            
            try:
                slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
                return slope
            except ZeroDivisionError:
                return 0
        
        def _calculate_performance_trend(self, metrics: List[PerformanceMetrics]) -> float:
            """Calculate overall performance trend (negative = degrading)."""
            if len(metrics) < 2:
                return 0
            
            # Calculate trends for key metrics
            latency_trend = self._calculate_trend([m.latency_ms for m in metrics])
            throughput_trend = self._calculate_trend([m.throughput_ops_per_sec for m in metrics])
            error_trend = self._calculate_trend([m.error_rate for m in metrics])
            
            # Combine trends (lower latency and errors = better, higher throughput = better)
            performance_trend = (throughput_trend - latency_trend - (error_trend * 1000)) / 3
            return performance_trend
        
        async def _auto_apply_optimization(self, task: OptimizationTask) -> None:
            """Auto-apply low-risk optimizations."""
            try:
                self.optimizer_metrics['auto_applied_optimizations'] += 1
                
                # Simulate applying optimization
                await asyncio.sleep(0.1)
                
                logger.info(f"Auto-applied optimization for {task.context.component_name}")
                
                # Send notification
                await self.queue_manager.enqueue(
                    'system.optimization_applied',
                    {
                        'task_id': task.task_id,
                        'component_id': task.context.component_id,
                        'component_name': task.context.component_name,
                        'optimization_type': task.result.optimization_type,
                        'auto_applied': True,
                        'timestamp': datetime.utcnow().isoformat()
                    },
                    domain='system'
                )
                
            except Exception as e:
                logger.error(f"Error auto-applying optimization: {e}")
        
        async def _send_optimization_result(self, task: OptimizationTask) -> None:
            """Send optimization result to the requesting conductor."""
            try:
                result_data = {
                    'task_id': task.task_id,
                    'component_id': task.context.component_id,
                    'component_name': task.context.component_name,
                    'component_type': task.context.component_type,
                    'status': task.status,
                    'recommendation': task.result.to_dict() if task.result else None,
                    'error': task.error,
                    'conductor_id': task.metadata.get('conductor_id') if hasattr(task, 'metadata') else None,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self.queue_manager.enqueue(
                    'system.optimization_results',
                    result_data,
                    domain='system'
                )
                
            except Exception as e:
                logger.error(f"Error sending optimization result: {e}")
        
        async def _system_analysis_worker(self) -> None:
            """Worker for system-wide optimization analysis."""
            logger.info("System analysis worker started")
            
            while self.running:
                try:
                    await asyncio.sleep(300)  # Run every 5 minutes
                    
                    # Analyze system-wide patterns
                    await self._analyze_system_patterns()
                    
                    # Identify cross-component optimizations
                    await self._identify_cross_component_optimizations()
                    
                    self.optimizer_metrics['multi_component_optimizations'] += 1
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in system analysis worker: {e}")
            
            logger.info("System analysis worker stopped")
        
        async def _predictive_optimization_worker(self) -> None:
            """Worker for predictive optimization analysis."""
            logger.info("Predictive optimization worker started")
            
            while self.running:
                try:
                    await asyncio.sleep(600)  # Run every 10 minutes
                    
                    # Predict future performance issues
                    await self._predict_performance_issues()
                    
                    self.optimizer_metrics['predictive_optimizations'] += 1
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in predictive optimization worker: {e}")
            
            logger.info("Predictive optimization worker stopped")
        
        async def _enhanced_metrics_processor(self) -> None:
            """Process enhanced metrics from conductors."""
            logger.info("Enhanced metrics processor started")
            
            while self.running:
                try:
                    # Process metrics from enhanced_metrics queue
                    # This would consume from system.enhanced_metrics queue
                    await asyncio.sleep(30)  # Process every 30 seconds
                    
                    # Update system-wide metrics
                    await self._update_system_metrics()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in metrics processor: {e}")
            
            logger.info("Enhanced metrics processor stopped")
        
        async def _analyze_system_patterns(self) -> None:
            """Analyze system-wide optimization patterns."""
            try:
                # Analyze patterns across all components
                component_types = {}
                for context in self.component_contexts.values():
                    comp_type = context.component_type
                    if comp_type not in component_types:
                        component_types[comp_type] = []
                    component_types[comp_type].append(context)
                
                # Find common patterns within component types
                for comp_type, contexts in component_types.items():
                    if len(contexts) > 1:
                        await self._find_common_patterns(comp_type, contexts)
                
            except Exception as e:
                logger.error(f"Error analyzing system patterns: {e}")
        
        async def _identify_cross_component_optimizations(self) -> None:
            """Identify optimizations that span multiple components."""
            try:
                # Analyze component dependencies and communication patterns
                # This would look for bottlenecks in component interactions
                pass
            except Exception as e:
                logger.error(f"Error identifying cross-component optimizations: {e}")
        
        async def _predict_performance_issues(self) -> None:
            """Predict future performance issues based on trends."""
            try:
                for context in self.component_contexts.values():
                    if len(context.metrics_history) > 20:
                        await self._predict_component_issues(context)
            except Exception as e:
                logger.error(f"Error predicting performance issues: {e}")
        
        async def _predict_component_issues(self, context: OptimizationContext) -> None:
            """Predict issues for a specific component."""
            try:
                recent_metrics = context.metrics_history[-20:]
                
                # Predict future latency
                latency_values = [m.latency_ms for m in recent_metrics]
                latency_trend = self._calculate_trend(latency_values)
                
                # If latency is increasing rapidly, predict an issue
                if latency_trend > 2.0:  # 2ms increase per measurement
                    await self._create_predictive_optimization_task(context, 'predicted_latency_issue')
                
                # Predict memory issues
                memory_values = [m.memory_usage_mb for m in recent_metrics]
                memory_trend = self._calculate_trend(memory_values)
                
                if memory_trend > 1.0:  # 1MB increase per measurement
                    await self._create_predictive_optimization_task(context, 'predicted_memory_issue')
                
            except Exception as e:
                logger.error(f"Error predicting issues for {context.component_id}: {e}")
        
        async def _create_predictive_optimization_task(self, context: OptimizationContext, issue_type: str) -> None:
            """Create a predictive optimization task."""
            try:
                task = OptimizationTask(
                    task_id=str(uuid.uuid4()),
                    context=context,
                    priority=3  # Medium priority for predictive tasks
                )
                
                task.metadata = {
                    'trigger_reason': 'predictive_analysis',
                    'predicted_issue': issue_type,
                    'urgent': False
                }
                
                await self.task_queue.put(task)
                self.optimization_tasks[task.task_id] = task
                
                self.optimizer_metrics['total_tasks'] += 1
                self.optimizer_metrics['active_tasks'] += 1
                
                logger.info(f"Created predictive optimization task for {context.component_name}: {issue_type}")
                
            except Exception as e:
                logger.error(f"Error creating predictive optimization task: {e}")
        
        async def _find_common_patterns(self, component_type: str, contexts: List[OptimizationContext]) -> None:
            """Find common optimization patterns within a component type."""
            try:
                # Analyze common issues across similar components
                common_issues = {}
                
                for context in contexts:
                    if len(context.metrics_history) > 10:
                        recent_metrics = context.metrics_history[-10:]
                        avg_metrics = self._calculate_metrics_average(recent_metrics)
                        
                        # Categorize performance characteristics
                        if avg_metrics.get('latency_ms', 0) > 100:
                            common_issues.setdefault('high_latency', []).append(context.component_id)
                        
                        if avg_metrics.get('error_rate', 0) > 0.02:
                            common_issues.setdefault('high_errors', []).append(context.component_id)
                        
                        if avg_metrics.get('memory_usage_mb', 0) > 500:
                            common_issues.setdefault('high_memory', []).append(context.component_id)
                
                # If multiple components have the same issue, create system-wide optimization
                for issue, component_ids in common_issues.items():
                    if len(component_ids) > 1:
                        await self._create_system_optimization_task(component_type, issue, component_ids)
                
            except Exception as e:
                logger.error(f"Error finding common patterns for {component_type}: {e}")
        
        async def _create_system_optimization_task(self, component_type: str, issue: str, component_ids: List[str]) -> None:
            """Create a system-wide optimization task."""
            try:
                # Create a meta-optimization task for multiple components
                logger.info(f"System-wide optimization needed for {component_type} components: {issue} affects {len(component_ids)} components")
                
                # This would create a more comprehensive optimization strategy
                # For now, just log the pattern
            except Exception as e:
                logger.error(f"Error creating system optimization task: {e}")
        
        async def _update_system_metrics(self) -> None:
            """Update system-wide metrics."""
            try:
                self.system_wide_metrics.update({
                    'total_components': len(self.component_contexts),
                    'active_optimizations': self.optimizer_metrics['active_tasks'],
                    'optimization_success_rate': (
                        self.optimizer_metrics['completed_tasks'] / 
                        max(self.optimizer_metrics['total_tasks'], 1)
                    ) * 100,
                    'avg_improvement_score': self.optimizer_metrics['quality_score_average'],
                    'last_updated': datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Error updating system metrics: {e}")
        
        def get_enhanced_optimizer_status(self) -> Dict[str, Any]:
            """Get comprehensive optimizer status."""
            return {
                'optimizer_metrics': self.optimizer_metrics.copy(),
                'system_metrics': self.system_wide_metrics.copy(),
                'config': self.config.copy(),
                'active_components': len(self.component_contexts),
                'queued_tasks': self.task_queue.qsize(),
                'worker_status': {
                    'optimization_workers': len(self.worker_tasks),
                    'analysis_workers': len(self.analysis_tasks)
                },
                'optimization_patterns': {k: len(v) for k, v in self.optimization_patterns.items()},
                'running': self.running,
                'security_enabled': self.security_enabled,
                'timestamp': datetime.utcnow().isoformat()
            }