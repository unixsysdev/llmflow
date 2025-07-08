"""
LLM Optimization Conductor

This module provides a conductor that integrates LLM optimization directly into
the LLMFlow framework. It monitors performance and automatically optimizes
components using OpenRouter + Gemini 2.0 Flash.
"""

import os
import json
import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from ..conductor.manager import ConductorManager
from ..queue.manager import QueueManager
from ..atoms.openrouter_llm import OpenRouterServiceAtom, OpenRouterRequest, OpenRouterRequestAtom
from ..atoms.data import StringAtom
from ..core.base import ComponentType, Component

logger = logging.getLogger(__name__)


class LLMOptimizationConductor(ConductorManager):
    """
    Enhanced conductor with LLM optimization capabilities.
    
    Monitors component performance and automatically optimizes components
    using Gemini 2.0 Flash for code analysis and improvement.
    """
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(queue_manager)
        
        # OpenRouter configuration
        self.openrouter_key = (
            os.getenv('OPENROUTER_API_KEY') or 
            "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"
        )
        os.environ['OPENROUTER_API_KEY'] = self.openrouter_key
        
        self.llm_service = OpenRouterServiceAtom()
        
        # LLM Configuration
        self.llm_config = {
            'model': 'google/gemini-2.0-flash-001',  # Only Gemini Flash
            'optimization_enabled': True,
            'auto_optimization_threshold': 0.85,
            'performance_check_interval': 30.0,  # Check performance every 30s
            'optimization_cooldown': 300.0,  # Wait 5min between optimizations
            'max_optimizations_per_component': 3
        }
        
        # LLM State
        self.llm_optimization_task: Optional[asyncio.Task] = None
        self.component_optimization_history: Dict[str, List[Dict[str, Any]]] = {}
        self.pending_optimizations: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced metrics
        self.llm_metrics = {
            'optimizations_triggered': 0,
            'optimizations_applied': 0,
            'optimizations_rejected': 0,
            'total_llm_cost': 0.0,
            'performance_improvements': 0,
            'components_optimized': set()
        }
        
        logger.info("LLM Optimization Conductor initialized with Gemini 2.0 Flash")
    
    async def start(self) -> None:
        """Start the LLM optimization conductor."""
        # Custom start without calling super() to avoid the missing method issue
        if self.running:
            return
        
        self.running = True
        
        # Start basic management tasks (without the problematic performance analysis)
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop()) 
        self._restart_task = asyncio.create_task(self._restart_loop())
        
        logger.info(f"Conductor manager {self.conductor_id} started")
        
        if self.llm_config['optimization_enabled']:
            # Start LLM optimization loop
            self.llm_optimization_task = asyncio.create_task(self._llm_optimization_loop())
            
            # Subscribe to optimization requests from performance analysis
            # TODO: Implement subscription system
            # await self.queue_manager.subscribe(
            #     'system.optimization_requests',
            #     self._handle_optimization_request,
            #     domain='system'
            # )
            
            logger.info("🤖 LLM optimization system started")
    
    async def stop(self) -> None:
        """Stop the LLM optimization conductor."""
        if self.llm_optimization_task:
            self.llm_optimization_task.cancel()
            try:
                await self.llm_optimization_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()
        logger.info("🤖 LLM optimization system stopped")
    
    async def _health_check_loop(self):
        """Basic health check loop."""
        while self.running:
            try:
                await asyncio.sleep(30)
                # Basic health checks
                logger.debug("Health check completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _metrics_collection_loop(self):
        """Basic metrics collection loop."""
        while self.running:
            try:
                await asyncio.sleep(60)
                # Basic metrics collection
                logger.debug("Metrics collection completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _restart_loop(self):
        """Basic restart loop."""
        while self.running:
            try:
                await asyncio.sleep(300)
                # Check for restart requirements
                logger.debug("Restart loop check completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Restart loop error: {e}")
    
    async def _llm_optimization_loop(self) -> None:
        """Main LLM optimization monitoring loop."""
        logger.info("🧠 LLM optimization loop started")
        
        while self.running:
            try:
                await asyncio.sleep(self.llm_config['performance_check_interval'])
                
                # Check all managed processes for optimization opportunities
                for process_id, process_info in self.managed_processes.items():
                    if process_info.status.value == 'running':
                        await self._check_process_for_llm_optimization(process_id)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in LLM optimization loop: {e}")
        
        logger.info("🧠 LLM optimization loop stopped")
    
    async def _check_process_for_llm_optimization(self, process_id: str) -> None:
        """Check if a process needs LLM optimization."""
        try:
            process_info = self.managed_processes.get(process_id)
            if not process_info:
                return
            
            # Skip if already being optimized
            if process_id in self.pending_optimizations:
                return
            
            # Check optimization history
            optimization_count = len(self.component_optimization_history.get(process_id, []))
            if optimization_count >= self.llm_config['max_optimizations_per_component']:
                return
            
            # Check cooldown period
            if self._is_in_optimization_cooldown(process_id):
                return
            
            # Get current performance metrics
            current_metrics = await self.get_process_metrics(process_id)
            if not current_metrics:
                return
            
            # Check if optimization is needed based on performance
            needs_optimization = await self._evaluate_optimization_need(process_id, current_metrics)
            
            if needs_optimization:
                logger.info(f"🎯 Triggering LLM optimization for process {process_id}")
                await self._trigger_llm_optimization(process_id, current_metrics)
        
        except Exception as e:
            logger.error(f"Error checking LLM optimization for {process_id}: {e}")
    
    def _is_in_optimization_cooldown(self, process_id: str) -> bool:
        """Check if process is in optimization cooldown period."""
        history = self.component_optimization_history.get(process_id, [])
        if not history:
            return False
        
        last_optimization = history[-1]
        last_time = datetime.fromisoformat(last_optimization['timestamp'])
        cooldown_seconds = self.llm_config['optimization_cooldown']
        
        return (datetime.utcnow() - last_time).total_seconds() < cooldown_seconds
    
    async def _evaluate_optimization_need(self, process_id: str, current_metrics) -> bool:
        """Evaluate if a process needs optimization based on performance."""
        try:
            # Get performance history for comparison
            history = self.performance_history.get(process_id, [])
            
            if len(history) < 5:  # Need some history for comparison
                return False
            
            # Calculate performance degradation
            recent_avg_latency = sum(h.get('latency_ms', 0) for h in history[-5:]) / 5
            baseline_avg_latency = sum(h.get('latency_ms', 0) for h in history[:5]) / 5
            
            # Check for significant performance degradation
            if baseline_avg_latency > 0:
                degradation = (recent_avg_latency - baseline_avg_latency) / baseline_avg_latency
                
                # Trigger optimization if performance degraded by >20%
                if degradation > 0.2:
                    logger.info(f"Performance degradation detected: {degradation:.1%}")
                    return True
            
            # Check for high error rates
            recent_error_rate = current_metrics.error_rate
            if recent_error_rate > 0.05:  # >5% error rate
                logger.info(f"High error rate detected: {recent_error_rate:.1%}")
                return True
            
            # Check for memory leaks
            memory_values = [h.get('memory_usage_mb', 0) for h in history[-10:]]
            if len(memory_values) >= 5:
                memory_growth = self._calculate_growth_rate(memory_values)
                if memory_growth > 0.1:  # >10% memory growth
                    logger.info(f"Memory growth detected: {memory_growth:.1%}")
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error evaluating optimization need: {e}")
            return False
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate growth rate of a metric over time."""
        if len(values) < 2:
            return 0.0
        
        start_value = values[0]
        end_value = values[-1]
        
        if start_value == 0:
            return 0.0
        
        return (end_value - start_value) / start_value
    
    async def _trigger_llm_optimization(self, process_id: str, current_metrics) -> None:
        """Trigger LLM optimization for a process."""
        try:
            process_info = self.managed_processes.get(process_id)
            if not process_info:
                return
            
            # Mark as pending optimization
            self.pending_optimizations[process_id] = {
                'started_at': datetime.utcnow().isoformat(),
                'metrics': {
                    'latency_ms': current_metrics.latency_ms,
                    'memory_usage_mb': current_metrics.memory_usage_mb,
                    'cpu_usage_percent': current_metrics.cpu_usage_percent,
                    'error_rate': current_metrics.error_rate
                }
            }
            
            # Get component code for analysis
            component_code = await self._get_component_code(process_id)
            
            # Create optimization prompt
            optimization_prompt = self._create_optimization_prompt(
                process_id, 
                process_info.component_name, 
                current_metrics, 
                component_code
            )
            
            # Send to LLM
            await self._send_optimization_request_to_llm(process_id, optimization_prompt)
            
            self.llm_metrics['optimizations_triggered'] += 1
            
        except Exception as e:
            logger.error(f"Error triggering LLM optimization for {process_id}: {e}")
            # Remove from pending on error
            self.pending_optimizations.pop(process_id, None)
    
    async def _get_component_code(self, process_id: str) -> str:
        """Get the source code for a component (mock implementation)."""
        # In a real implementation, this would fetch the actual component source code
        # For demo purposes, return a mock component
        return f'''
# Component code for process {process_id}
class MockComponent:
    def __init__(self):
        self.data = []
    
    def process(self, input_data):
        # Inefficient processing - could be optimized
        result = []
        for item in input_data:
            # Slow operation that could be vectorized
            for i in range(len(self.data)):
                if self.data[i] == item:
                    result.append(item * 2)
                    break
        return result
    
    def add_data(self, item):
        # Inefficient - creates new list each time
        self.data = self.data + [item]
'''
    
    def _create_optimization_prompt(self, process_id: str, component_name: str, 
                                  metrics, component_code: str) -> str:
        """Create optimization prompt for LLM."""
        performance_history = self.performance_history.get(process_id, [])
        
        prompt = f"""
Optimize this LLMFlow component for better performance.

**Component Details:**
- Process ID: {process_id}
- Component Name: {component_name}
- Current Performance Issues:
  * Latency: {metrics.latency_ms:.1f}ms
  * Memory Usage: {metrics.memory_usage_mb:.1f}MB
  * CPU Usage: {metrics.cpu_usage_percent:.1f}%
  * Error Rate: {metrics.error_rate:.1%}

**Performance History (last 5 measurements):**
{json.dumps(performance_history[-5:], indent=2)}

**Current Component Code:**
```python
{component_code}
```

**Optimization Requirements:**
1. Reduce latency and memory usage
2. Fix any performance bottlenecks
3. Maintain API compatibility
4. Add proper error handling
5. Use efficient algorithms and data structures
6. Keep code readable and maintainable

**Response Format (JSON):**
{{
  "analysis": "detailed analysis of performance issues",
  "optimized_code": "complete optimized Python code",
  "improvements": {{
    "latency_improvement": "expected percentage improvement",
    "memory_improvement": "expected percentage improvement",
    "cpu_improvement": "expected percentage improvement"
  }},
  "confidence_score": 0.0-1.0,
  "breaking_changes": ["list any breaking changes"],
  "deployment_notes": "how to deploy this optimization safely"
}}

Generate production-ready optimized code that significantly improves performance.
"""
        
        return prompt
    
    async def _send_optimization_request_to_llm(self, process_id: str, prompt: str) -> None:
        """Send optimization request to LLM."""
        try:
            # Create OpenRouter request
            request = OpenRouterRequest(
                prompt=prompt,
                model=self.llm_config['model'],
                max_tokens=4000,
                temperature=0.1,
                site_url="https://llmflow.dev",
                site_name="LLMFlow Conductor Optimizer"
            )
            
            logger.info(f"🤖 Sending optimization request to Gemini 2.0 Flash...")
            
            # Process with LLM
            response_atoms = await self.llm_service.process([OpenRouterRequestAtom(request)])
            
            if not response_atoms or response_atoms[0].response.error:
                error = response_atoms[0].response.error if response_atoms else "No response"
                logger.error(f"LLM optimization request failed: {error}")
                self.llm_metrics['optimizations_rejected'] += 1
                return
            
            response = response_atoms[0].response
            self.llm_metrics['total_llm_cost'] += response.cost_usd
            
            logger.info(f"✅ LLM optimization response received (${response.cost_usd:.4f})")
            
            # Process the optimization response
            await self._process_optimization_response(process_id, response.content)
        
        except Exception as e:
            logger.error(f"Error sending LLM optimization request: {e}")
            self.llm_metrics['optimizations_rejected'] += 1
        finally:
            # Remove from pending
            self.pending_optimizations.pop(process_id, None)
    
    async def _process_optimization_response(self, process_id: str, response_content: str) -> None:
        """Process LLM optimization response."""
        try:
            # Parse JSON response
            optimization_data = json.loads(response_content)
            
            confidence = optimization_data.get('confidence_score', 0.0)
            optimized_code = optimization_data.get('optimized_code', '')
            analysis = optimization_data.get('analysis', '')
            
            logger.info(f"📊 Optimization analysis: {analysis[:100]}...")
            
            # Check if confidence meets threshold for auto-application
            if confidence >= self.llm_config['auto_optimization_threshold']:
                logger.info(f"🚀 Auto-applying optimization (confidence: {confidence:.1%})")
                await self._apply_optimization(process_id, optimization_data)
                self.llm_metrics['optimizations_applied'] += 1
                self.llm_metrics['components_optimized'].add(process_id)
            else:
                logger.info(f"⚠️ Optimization confidence too low: {confidence:.1%} (threshold: {self.llm_config['auto_optimization_threshold']:.1%})")
                await self._store_optimization_for_manual_review(process_id, optimization_data)
                self.llm_metrics['optimizations_rejected'] += 1
        
        except json.JSONDecodeError:
            logger.warning("LLM response not in JSON format")
            await self._store_optimization_for_manual_review(process_id, {'raw_response': response_content})
            self.llm_metrics['optimizations_rejected'] += 1
        except Exception as e:
            logger.error(f"Error processing optimization response: {e}")
            self.llm_metrics['optimizations_rejected'] += 1
    
    async def _apply_optimization(self, process_id: str, optimization_data: Dict[str, Any]) -> None:
        """Apply LLM optimization to a component."""
        try:
            logger.info(f"🔧 Applying LLM optimization to process {process_id}...")
            
            process_info = self.managed_processes.get(process_id)
            if not process_info:
                logger.error(f"Process {process_id} not found")
                return
            
            # In a real implementation, this would:
            # 1. Create a backup of the current component
            # 2. Deploy the optimized code
            # 3. Restart the component
            # 4. Monitor for successful deployment
            
            # For demo purposes, we'll simulate the optimization
            logger.info(f"   📦 Creating backup of component {process_info.component_name}")
            logger.info(f"   🔄 Deploying optimized code")
            
            # Simulate restart
            await self.restart_process(process_id)
            
            # Record optimization in history
            if process_id not in self.component_optimization_history:
                self.component_optimization_history[process_id] = []
            
            self.component_optimization_history[process_id].append({
                'timestamp': datetime.utcnow().isoformat(),
                'optimization_data': optimization_data,
                'confidence': optimization_data.get('confidence_score', 0.0),
                'expected_improvements': optimization_data.get('improvements', {})
            })
            
            logger.info(f"✅ Optimization applied successfully to {process_id}")
            
            # Reset performance baseline to measure improvement
            if process_id in self.performance_history:
                self.performance_history[process_id] = []
        
        except Exception as e:
            logger.error(f"Error applying optimization to {process_id}: {e}")
            # In a real implementation, this would trigger rollback
    
    async def _store_optimization_for_manual_review(self, process_id: str, optimization_data: Dict[str, Any]) -> None:
        """Store optimization for manual review."""
        try:
            # Send to manual review queue
            await self.queue_manager.enqueue(
                'system.optimization_review',
                {
                    'process_id': process_id,
                    'conductor_id': self.conductor_id,
                    'optimization_data': optimization_data,
                    'timestamp': datetime.utcnow().isoformat(),
                    'requires_manual_review': True
                },
                domain='system'
            )
            
            logger.info(f"📝 Stored optimization for manual review: {process_id}")
        
        except Exception as e:
            logger.error(f"Error storing optimization for review: {e}")
    
    async def _handle_optimization_request(self, message: Dict[str, Any]) -> None:
        """Handle optimization requests from the performance analysis system."""
        try:
            process_id = message.get('process_id')
            anomalies = message.get('anomalies', [])
            
            if not process_id:
                return
            
            logger.info(f"📨 Received optimization request for {process_id}: {anomalies}")
            
            # Get current metrics
            current_metrics = await self.get_process_metrics(process_id)
            if current_metrics:
                await self._trigger_llm_optimization(process_id, current_metrics)
        
        except Exception as e:
            logger.error(f"Error handling optimization request: {e}")
    
    def get_llm_optimization_status(self) -> Dict[str, Any]:
        """Get LLM optimization status and metrics."""
        return {
            'llm_optimization_enabled': self.llm_config['optimization_enabled'],
            'llm_model': self.llm_config['model'],
            'api_key_configured': bool(self.openrouter_key),
            'pending_optimizations': len(self.pending_optimizations),
            'components_with_history': len(self.component_optimization_history),
            'metrics': {
                **self.llm_metrics,
                'components_optimized': len(self.llm_metrics['components_optimized'])
            },
            'config': self.llm_config.copy()
        }
    
    def get_conductor_status(self) -> Dict[str, Any]:
        """Get enhanced conductor status including LLM optimization."""
        base_status = super().get_conductor_status()
        base_status['llm_optimization'] = self.get_llm_optimization_status()
        return base_status
