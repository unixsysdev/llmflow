"""
LLM-Powered Optimization Molecules

This module provides advanced optimization molecules that use OpenAI's LLM
for intelligent code analysis and optimization recommendations.
"""

import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..atoms.base import ServiceAtom, DataAtom
from ..atoms.data import StringAtom, BooleanAtom
from ..atoms.llm import (
    LLMRequest, LLMResponse, LLMRequestAtom, LLMResponseAtom, OpenAIServiceAtom
)
from ..queue.manager import QueueManager
from .optimization import (
    PerformanceMetrics, OptimizationRecommendation,
    PerformanceMetricsAtom, OptimizationRecommendationAtom
)

logger = logging.getLogger(__name__)


class LLMCodeAnalysisMolecule(ServiceAtom):
    """Advanced code analysis using LLM."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="llm_code_analysis_molecule",
            input_types=[
                "llmflow.atoms.data.StringAtom",  # Source code
                "llmflow.atoms.data.StringAtom"   # Performance metrics JSON
            ],
            output_types=[
                "llmflow.atoms.data.StringAtom",  # Analysis report
                "llmflow.atoms.data.BooleanAtom"  # Issues found
            ]
        )
        self.queue_manager = queue_manager
        self.openai_service = OpenAIServiceAtom()
        
        # Analysis prompt templates
        self.analysis_prompts = {
            'system': """You are an expert software engineer and performance optimization specialist. 
Analyze the provided code and performance metrics to identify optimization opportunities.

Provide detailed, actionable recommendations focusing on:
1. Performance bottlenecks and inefficiencies
2. Memory usage optimizations
3. Algorithmic improvements
4. Code structure and maintainability
5. Potential bugs or error-prone patterns

Be specific, technical, and provide concrete examples where possible.""",
            
            'user': """Please analyze this code and its performance metrics:

=== CODE ===
{code}

=== PERFORMANCE METRICS ===
{metrics}

=== ANALYSIS REQUEST ===
Provide a comprehensive analysis in JSON format with the following structure:
{{
    "issues_found": true/false,
    "primary_concerns": ["list of main issues"],
    "optimization_opportunities": [
        {{
            "type": "latency|memory|throughput|reliability",
            "severity": "low|medium|high|critical",
            "description": "detailed description",
            "code_location": "specific code section if applicable",
            "recommendation": "specific action to take"
        }}
    ],
    "code_quality_score": 0-100,
    "estimated_improvement_potential": 0.0-1.0,
    "implementation_complexity": "low|medium|high",
    "risk_assessment": "low|medium|high"
}}

Focus on practical, implementable optimizations that will have measurable impact."""
        }
    
    async def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Process code analysis using LLM."""
        if len(inputs) < 2:
            return [StringAtom("Insufficient inputs"), BooleanAtom(False)]
        
        code_atom = inputs[0]
        metrics_atom = inputs[1]
        
        if not isinstance(code_atom, StringAtom) or not isinstance(metrics_atom, StringAtom):
            return [StringAtom("Invalid input types"), BooleanAtom(False)]
        
        try:
            # Prepare LLM request
            user_prompt = self.analysis_prompts['user'].format(
                code=code_atom.value,
                metrics=metrics_atom.value
            )
            
            llm_request = LLMRequest(
                prompt=user_prompt,
                system_prompt=self.analysis_prompts['system'],
                model="gpt-4",
                max_tokens=3000,
                temperature=0.1,
                metadata={'analysis_type': 'code_performance'}
            )
            
            # Make LLM request
            llm_response_atoms = await self.openai_service.process([LLMRequestAtom(llm_request)])
            
            if not llm_response_atoms or llm_response_atoms[0].response.error:
                error_msg = llm_response_atoms[0].response.error if llm_response_atoms else "No response"
                logger.error(f"LLM analysis failed: {error_msg}")
                return [StringAtom(f"Analysis failed: {error_msg}"), BooleanAtom(False)]
            
            analysis_content = llm_response_atoms[0].response.content
            
            # Try to parse JSON response
            try:
                analysis_json = json.loads(analysis_content)
                issues_found = analysis_json.get('issues_found', False)
                
                logger.info(f"LLM code analysis completed, issues found: {issues_found}")
                return [StringAtom(analysis_content), BooleanAtom(issues_found)]
                
            except json.JSONDecodeError:
                # If JSON parsing fails, still return the raw analysis
                logger.warning("LLM response not in JSON format, returning raw content")
                return [StringAtom(analysis_content), BooleanAtom(True)]
        
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return [StringAtom(f"Analysis error: {str(e)}"), BooleanAtom(False)]


class LLMOptimizationGeneratorMolecule(ServiceAtom):
    """Generate optimization code using LLM."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="llm_optimization_generator_molecule",
            input_types=[
                "llmflow.atoms.data.StringAtom",  # Analysis report
                "llmflow.atoms.data.StringAtom",  # Original code
                "llmflow.atoms.data.StringAtom"   # Optimization type
            ],
            output_types=[
                "llmflow.molecules.optimization.OptimizationRecommendationAtom"
            ]
        )
        self.queue_manager = queue_manager
        self.openai_service = OpenAIServiceAtom()
        
        # Optimization prompt templates
        self.optimization_prompts = {
            'system': """You are a senior software engineer specializing in performance optimization. 
Generate concrete, implementable code optimizations based on analysis reports.

Your optimizations should be:
1. Safe and backwards-compatible when possible
2. Well-documented with clear explanations
3. Measurable in terms of performance impact
4. Production-ready with proper error handling

Provide both the optimized code and rollback instructions.""",
            
            'user': """Based on this analysis and code, generate specific optimization recommendations:

=== ANALYSIS REPORT ===
{analysis}

=== ORIGINAL CODE ===
{code}

=== OPTIMIZATION TYPE ===
{optimization_type}

Generate a detailed optimization recommendation in JSON format:
{{
    "recommendation_id": "unique_id",
    "target_component": "component_name",
    "optimization_type": "{optimization_type}",
    "description": "clear description of the optimization",
    "expected_improvement": 0.0-1.0,
    "confidence_score": 0.0-1.0,
    "implementation_code": "complete optimized code",
    "rollback_code": "code to revert changes",
    "test_strategy": "how to validate the optimization",
    "deployment_steps": ["step1", "step2", "..."],
    "risk_assessment": {{
        "level": "low|medium|high",
        "factors": ["list of risk factors"],
        "mitigation": ["mitigation strategies"]
    }},
    "metrics_to_monitor": ["metric1", "metric2", "..."],
    "estimated_effort_hours": integer,
    "prerequisites": ["any prerequisites"],
    "metadata": {{
        "analysis_timestamp": "timestamp",
        "optimization_approach": "description of approach",
        "alternative_approaches": ["other possible approaches"]
    }}
}}

Make the optimization specific, actionable, and production-ready."""
        }
    
    async def process(self, inputs: List[DataAtom]) -> List[OptimizationRecommendationAtom]:
        """Generate optimization recommendations using LLM."""
        if len(inputs) < 3:
            return [self._create_error_recommendation("Insufficient inputs")]
        
        analysis_atom = inputs[0]
        code_atom = inputs[1]
        optimization_type_atom = inputs[2]
        
        if not all(isinstance(atom, StringAtom) for atom in inputs):
            return [self._create_error_recommendation("Invalid input types")]
        
        try:
            # Prepare LLM request
            user_prompt = self.optimization_prompts['user'].format(
                analysis=analysis_atom.value,
                code=code_atom.value,
                optimization_type=optimization_type_atom.value
            )
            
            llm_request = LLMRequest(
                prompt=user_prompt,
                system_prompt=self.optimization_prompts['system'],
                model="gpt-4",
                max_tokens=4000,
                temperature=0.1,
                metadata={'optimization_type': optimization_type_atom.value}
            )
            
            # Make LLM request
            llm_response_atoms = await self.openai_service.process([LLMRequestAtom(llm_request)])
            
            if not llm_response_atoms or llm_response_atoms[0].response.error:
                error_msg = llm_response_atoms[0].response.error if llm_response_atoms else "No response"
                return [self._create_error_recommendation(f"LLM request failed: {error_msg}")]
            
            optimization_content = llm_response_atoms[0].response.content
            
            # Parse LLM response
            recommendation = await self._parse_optimization_response(
                optimization_content, 
                optimization_type_atom.value
            )
            
            logger.info(f"Generated optimization recommendation: {recommendation.recommendation_id}")
            return [OptimizationRecommendationAtom(recommendation)]
        
        except Exception as e:
            logger.error(f"Optimization generation failed: {e}")
            return [self._create_error_recommendation(f"Generation error: {str(e)}")]
    
    async def _parse_optimization_response(self, content: str, optimization_type: str) -> OptimizationRecommendation:
        """Parse LLM response into OptimizationRecommendation."""
        try:
            # Try to parse as JSON
            recommendation_data = json.loads(content)
            
            return OptimizationRecommendation(
                recommendation_id=recommendation_data.get('recommendation_id', str(uuid.uuid4())),
                target_component=recommendation_data.get('target_component', 'unknown'),
                optimization_type=recommendation_data.get('optimization_type', optimization_type),
                description=recommendation_data.get('description', 'LLM-generated optimization'),
                expected_improvement=float(recommendation_data.get('expected_improvement', 0.2)),
                confidence_score=float(recommendation_data.get('confidence_score', 0.7)),
                implementation_code=recommendation_data.get('implementation_code', ''),
                rollback_code=recommendation_data.get('rollback_code', ''),
                metadata=recommendation_data.get('metadata', {
                    'llm_generated': True,
                    'created_at': datetime.utcnow().isoformat()
                })
            )
        
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic recommendation
            logger.warning("Failed to parse LLM response as JSON, creating basic recommendation")
            
            return OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                target_component='unknown',
                optimization_type=optimization_type,
                description=f"LLM-generated {optimization_type} optimization",
                expected_improvement=0.2,
                confidence_score=0.5,
                implementation_code=content,  # Use raw content as implementation
                rollback_code="# Restore original implementation",
                metadata={
                    'llm_generated': True,
                    'raw_response': True,
                    'created_at': datetime.utcnow().isoformat()
                }
            )
    
    def _create_error_recommendation(self, error_message: str) -> OptimizationRecommendationAtom:
        """Create an error recommendation."""
        error_recommendation = OptimizationRecommendation(
            recommendation_id=str(uuid.uuid4()),
            target_component='unknown',
            optimization_type='error',
            description=f"Optimization generation failed: {error_message}",
            expected_improvement=0.0,
            confidence_score=0.0,
            implementation_code='',
            rollback_code='',
            metadata={
                'error': True,
                'error_message': error_message,
                'created_at': datetime.utcnow().isoformat()
            }
        )
        return OptimizationRecommendationAtom(error_recommendation)


class LLMSystemOptimizationMolecule(ServiceAtom):
    """System-wide optimization analysis using LLM."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="llm_system_optimization_molecule",
            input_types=[
                "llmflow.atoms.data.StringAtom"  # System metrics and components JSON
            ],
            output_types=[
                "llmflow.atoms.data.StringAtom"  # System optimization recommendations
            ]
        )
        self.queue_manager = queue_manager
        self.openai_service = OpenAIServiceAtom()
        
        self.system_analysis_prompt = {
            'system': """You are a systems architect specializing in distributed systems optimization.
Analyze system-wide performance patterns and recommend architectural improvements.

Focus on:
1. Component interaction bottlenecks
2. Resource allocation optimization
3. Queue and communication patterns
4. Load balancing and scaling opportunities
5. System reliability and fault tolerance""",
            
            'user': """Analyze this system-wide performance data and provide optimization recommendations:

=== SYSTEM DATA ===
{system_data}

Provide recommendations in JSON format:
{{
    "system_health_score": 0-100,
    "critical_issues": ["list of critical system issues"],
    "optimization_priorities": [
        {{
            "priority": 1-10,
            "area": "component_name or system_area",
            "issue": "description of issue",
            "recommendation": "specific recommendation",
            "estimated_impact": 0.0-1.0,
            "implementation_effort": "low|medium|high"
        }}
    ],
    "architectural_recommendations": ["list of architectural changes"],
    "immediate_actions": ["urgent actions needed"],
    "long_term_strategy": "overall system optimization strategy"
}}"""
        }
    
    async def process(self, inputs: List[DataAtom]) -> List[StringAtom]:
        """Process system-wide optimization analysis."""
        if not inputs or not isinstance(inputs[0], StringAtom):
            return [StringAtom("Invalid input for system analysis")]
        
        system_data = inputs[0].value
        
        try:
            user_prompt = self.system_analysis_prompt['user'].format(system_data=system_data)
            
            llm_request = LLMRequest(
                prompt=user_prompt,
                system_prompt=self.system_analysis_prompt['system'],
                model="gpt-4",
                max_tokens=3000,
                temperature=0.1,
                metadata={'analysis_type': 'system_optimization'}
            )
            
            llm_response_atoms = await self.openai_service.process([LLMRequestAtom(llm_request)])
            
            if llm_response_atoms and not llm_response_atoms[0].response.error:
                analysis_content = llm_response_atoms[0].response.content
                logger.info("System optimization analysis completed")
                return [StringAtom(analysis_content)]
            else:
                error_msg = llm_response_atoms[0].response.error if llm_response_atoms else "No response"
                logger.error(f"System analysis failed: {error_msg}")
                return [StringAtom(f"System analysis failed: {error_msg}")]
        
        except Exception as e:
            logger.error(f"System optimization analysis error: {e}")
            return [StringAtom(f"Analysis error: {str(e)}")]
