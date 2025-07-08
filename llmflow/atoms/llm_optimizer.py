"""
LLM-Powered Component Optimizer

This module provides automated component optimization using LLM analysis and code generation.
It can analyze existing components, generate optimized versions, and deploy them automatically.
"""

import os
import json
import logging
import asyncio
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from ..core.base import ServiceAtom, DataAtom, ValidationResult
from ..atoms.data import StringAtom, BooleanAtom
from ..atoms.openrouter_llm import (
    OpenRouterRequest, OpenRouterResponse, 
    OpenRouterRequestAtom, OpenRouterResponseAtom, 
    OpenRouterServiceAtom
)
from ..queue.manager import QueueManager

logger = logging.getLogger(__name__)


class ComponentAnalysisAtom(DataAtom):
    """Data atom for component analysis results."""
    
    def __init__(self, analysis_data: Dict[str, Any]):
        super().__init__(analysis_data)
        self.analysis_data = analysis_data
    
    def validate(self) -> ValidationResult:
        """Validate the analysis data."""
        errors = []
        
        required_fields = ['component_name', 'current_code', 'performance_metrics', 'analysis_timestamp']
        for field in required_fields:
            if field not in self.analysis_data:
                errors.append(f"Missing required field: {field}")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def serialize(self) -> bytes:
        return json.dumps(self.value).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'ComponentAnalysisAtom':
        analysis_data = json.loads(data.decode('utf-8'))
        return cls(analysis_data)


class OptimizedComponentAtom(DataAtom):
    """Data atom for optimized component code."""
    
    def __init__(self, component_data: Dict[str, Any]):
        super().__init__(component_data)
        self.component_data = component_data
    
    def validate(self) -> ValidationResult:
        """Validate the optimized component data."""
        errors = []
        
        required_fields = ['component_name', 'optimized_code', 'optimization_type', 'confidence_score']
        for field in required_fields:
            if field not in self.component_data:
                errors.append(f"Missing required field: {field}")
        
        if 'confidence_score' in self.component_data:
            score = self.component_data['confidence_score']
            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                errors.append("Confidence score must be between 0.0 and 1.0")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def serialize(self) -> bytes:
        return json.dumps(self.value).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'OptimizedComponentAtom':
        component_data = json.loads(data.decode('utf-8'))
        return cls(component_data)


class LLMComponentOptimizerAtom(ServiceAtom):
    """Advanced LLM-powered component optimizer that can actually implement optimizations."""
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="llm_component_optimizer",
            input_types=["llmflow.atoms.llm_optimizer.ComponentAnalysisAtom"],
            output_types=["llmflow.atoms.llm_optimizer.OptimizedComponentAtom"]
        )
        
        self.queue_manager = queue_manager
        self.openrouter_service = OpenRouterServiceAtom()
        
        # Get OpenRouter API key from environment
        self.api_key = os.getenv('OPENROUTER_API_KEY') or "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"
        
        # Optimization prompts optimized for code generation
        self.optimization_prompts = {
            'system': """You are an expert software engineer specializing in performance optimization and modern software architecture. 

Your task is to analyze components and generate optimized, production-ready code that:
1. Maintains API compatibility while improving performance
2. Uses modern best practices and design patterns
3. Includes proper error handling and logging
4. Is well-documented and testable
5. Follows the existing codebase patterns

When optimizing:
- Focus on measurable performance improvements
- Maintain backward compatibility when possible
- Use appropriate data structures and algorithms
- Implement proper resource management
- Add comprehensive error handling
- Include meaningful logging and metrics

Always provide complete, runnable code that can be directly used.""",
            
            'analysis': """Analyze this component and generate an optimized version:

=== COMPONENT ANALYSIS ===
Component: {component_name}
Current Performance Metrics: {performance_metrics}

=== CURRENT CODE ===
{current_code}

=== OPTIMIZATION REQUIREMENTS ===
Optimization Type: {optimization_type}
Target Improvement: {target_improvement}%
Constraints: {constraints}

=== TASK ===
Generate a complete, optimized version of this component. Provide your response in JSON format:

{{
    "analysis": {{
        "identified_bottlenecks": ["list of performance issues found"],
        "optimization_opportunities": ["specific improvements possible"],
        "estimated_improvement": "percentage improvement expected",
        "risk_assessment": "low|medium|high"
    }},
    "optimized_component": {{
        "component_name": "{component_name}",
        "optimized_code": "COMPLETE OPTIMIZED CODE HERE",
        "optimization_explanation": "detailed explanation of changes made",
        "breaking_changes": ["list any breaking changes"],
        "migration_notes": "how to migrate from old to new version",
        "test_cases": "suggested test cases for validation",
        "performance_expectations": "expected performance improvements"
    }},
    "implementation": {{
        "deployment_steps": ["step by step deployment guide"],
        "rollback_plan": "how to rollback if issues occur",
        "monitoring_metrics": ["metrics to monitor after deployment"],
        "validation_checklist": ["items to verify after deployment"]
    }},
    "confidence_score": 0.0-1.0
}}

Ensure the optimized_code field contains complete, runnable Python code that can be directly used.""",
            
            'graph_app_generation': """Generate a complete clock application using graph-based data flow architecture:

=== REQUIREMENTS ===
- Build a real-time clock application using LLMFlow's graph-based architecture
- Use atoms, molecules, and conductors for data flow
- Implement proper data streaming for real-time updates
- Create a simple but functional visual interface
- Use the queue-based communication pattern
- Make it easily extensible for future enhancements

=== ARCHITECTURE REQUIREMENTS ===
- Data Atoms: Time data types (TimeAtom, ClockStateAtom)
- Service Atoms: Time formatting, timezone conversion, display rendering
- Molecules: Clock logic composition, UI update management
- Conductors: Runtime management and data flow orchestration
- Queue-based communication throughout

Generate complete, working code for all components. Respond in JSON format:

{{
    "application": {{
        "name": "LLMFlow Clock App",
        "description": "Graph-based real-time clock application",
        "architecture": "distributed queue-based with data flow graph"
    }},
    "components": {{
        "atoms": {{
            "TimeAtom": "complete code for time data atom",
            "ClockStateAtom": "complete code for clock state atom",
            "FormatTimeServiceAtom": "time formatting service",
            "TimezoneServiceAtom": "timezone conversion service",
            "DisplayServiceAtom": "display rendering service"
        }},
        "molecules": {{
            "ClockLogicMolecule": "complete clock logic composition",
            "UIUpdateMolecule": "UI update management",
            "TimezoneManagerMolecule": "timezone management"
        }},
        "conductors": {{
            "ClockConductor": "main clock conductor for runtime management"
        }},
        "main_app": "complete main application file"
    }},
    "deployment": {{
        "requirements": ["list of dependencies"],
        "setup_instructions": ["step by step setup"],
        "run_instructions": "how to run the application"
    }},
    "confidence_score": 0.95
}}

Make this a fully functional, production-ready clock application that demonstrates LLMFlow's capabilities."""
        }
        
        # Statistics tracking
        self.stats = {
            'optimizations_performed': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'total_confidence_score': 0.0,
            'optimization_types': {},
            'generated_components': []
        }
    
    async def process(self, inputs: List[DataAtom]) -> List[OptimizedComponentAtom]:
        """Process component optimization request."""
        if not inputs or not isinstance(inputs[0], ComponentAnalysisAtom):
            return [self._create_error_component("Invalid input")]
        
        analysis_atom = inputs[0]
        validation = analysis_atom.validate()
        
        if not validation.is_valid:
            return [self._create_error_component(f"Validation failed: {validation.errors}")]
        
        self.stats['optimizations_performed'] += 1
        
        try:
            # Determine if this is a special app generation request
            if analysis_atom.analysis_data.get('component_name') == 'clock_app_generator':
                return await self._generate_clock_app(analysis_atom.analysis_data)
            else:
                return await self._optimize_component(analysis_atom.analysis_data)
        
        except Exception as e:
            self.stats['failed_optimizations'] += 1
            logger.error(f"Component optimization failed: {e}")
            return [self._create_error_component(f"Optimization error: {str(e)}")]
    
    async def _generate_clock_app(self, analysis_data: Dict[str, Any]) -> List[OptimizedComponentAtom]:
        """Generate the complete graph-based clock application."""
        logger.info("Generating complete clock application using LLM...")
        
        try:
            # Use the specialized graph app generation prompt
            openrouter_request = OpenRouterRequest(
                prompt=self.optimization_prompts['graph_app_generation'],
                system_prompt=self.optimization_prompts['system'],
                model="google/gemini-2.0-flash-001",  # Use the best model for code generation
                max_tokens=8000,
                temperature=0.1,
                site_url="https://llmflow.dev",
                site_name="LLMFlow Clock App Generator",
                metadata={'generation_type': 'complete_application'}
            )
            
            # Make the request
            response_atoms = await self.openrouter_service.process([OpenRouterRequestAtom(openrouter_request)])
            
            if not response_atoms or response_atoms[0].response.error:
                error_msg = response_atoms[0].response.error if response_atoms else "No response"
                return [self._create_error_component(f"Clock app generation failed: {error_msg}")]
            
            response_content = response_atoms[0].response.content
            
            # Parse the response and create component files
            try:
                clock_app_data = json.loads(response_content)
                
                # Create the optimized component with complete app
                optimized_component = {
                    'component_name': 'llmflow_clock_app',
                    'optimization_type': 'complete_application_generation',
                    'optimized_code': self._format_clock_app_code(clock_app_data),
                    'confidence_score': clock_app_data.get('confidence_score', 0.95),
                    'metadata': {
                        'generation_type': 'complete_application',
                        'model_used': 'google/gemini-2.0-flash-001',
                        'tokens_used': response_atoms[0].response.usage_tokens,
                        'cost_usd': response_atoms[0].response.cost_usd,
                        'components_generated': list(clock_app_data.get('components', {}).keys()),
                        'timestamp': datetime.now().isoformat()
                    },
                    'deployment_info': clock_app_data.get('deployment', {}),
                    'app_structure': clock_app_data
                }
                
                # Actually write the files
                await self._write_clock_app_files(clock_app_data)
                
                self.stats['successful_optimizations'] += 1
                self.stats['total_confidence_score'] += optimized_component['confidence_score']
                self.stats['generated_components'].append('llmflow_clock_app')
                
                logger.info("Clock application generated successfully!")
                return [OptimizedComponentAtom(optimized_component)]
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                # Return raw response as code
                optimized_component = {
                    'component_name': 'llmflow_clock_app',
                    'optimization_type': 'complete_application_generation',
                    'optimized_code': response_content,
                    'confidence_score': 0.7,
                    'metadata': {
                        'generation_type': 'raw_response',
                        'parsing_failed': True,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                return [OptimizedComponentAtom(optimized_component)]
        
        except Exception as e:
            logger.error(f"Clock app generation failed: {e}")
            return [self._create_error_component(f"Generation error: {str(e)}")]
    
    async def _optimize_component(self, analysis_data: Dict[str, Any]) -> List[OptimizedComponentAtom]:
        """Optimize a specific component."""
        component_name = analysis_data['component_name']
        current_code = analysis_data['current_code']
        performance_metrics = analysis_data['performance_metrics']
        optimization_type = analysis_data.get('optimization_type', 'performance')
        target_improvement = analysis_data.get('target_improvement', 20)
        constraints = analysis_data.get('constraints', 'maintain API compatibility')
        
        try:
            # Prepare the optimization request
            user_prompt = self.optimization_prompts['analysis'].format(
                component_name=component_name,
                performance_metrics=json.dumps(performance_metrics, indent=2),
                current_code=current_code,
                optimization_type=optimization_type,
                target_improvement=target_improvement,
                constraints=constraints
            )
            
            openrouter_request = OpenRouterRequest(
                prompt=user_prompt,
                system_prompt=self.optimization_prompts['system'],
                model="google/gemini-2.0-flash-001",
                max_tokens=6000,
                temperature=0.1,
                site_url="https://llmflow.dev",
                site_name="LLMFlow Component Optimizer",
                metadata={'optimization_type': optimization_type, 'component': component_name}
            )
            
            # Make the request
            response_atoms = await self.openrouter_service.process([OpenRouterRequestAtom(openrouter_request)])
            
            if not response_atoms or response_atoms[0].response.error:
                error_msg = response_atoms[0].response.error if response_atoms else "No response"
                return [self._create_error_component(f"Optimization failed: {error_msg}")]
            
            response_content = response_atoms[0].response.content
            
            # Parse the optimization response
            optimization_result = await self._parse_optimization_response(response_content, analysis_data)
            
            # Track optimization type
            if optimization_type not in self.stats['optimization_types']:
                self.stats['optimization_types'][optimization_type] = 0
            self.stats['optimization_types'][optimization_type] += 1
            
            self.stats['successful_optimizations'] += 1
            self.stats['total_confidence_score'] += optimization_result['confidence_score']
            
            logger.info(f"Component {component_name} optimized successfully")
            return [OptimizedComponentAtom(optimization_result)]
        
        except Exception as e:
            logger.error(f"Component optimization failed: {e}")
            return [self._create_error_component(f"Optimization error: {str(e)}")]
    
    async def _parse_optimization_response(self, content: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the LLM optimization response."""
        try:
            optimization_data = json.loads(content)
            
            optimized_component = optimization_data.get('optimized_component', {})
            
            return {
                'component_name': optimized_component.get('component_name', analysis_data['component_name']),
                'optimization_type': analysis_data.get('optimization_type', 'performance'),
                'optimized_code': optimized_component.get('optimized_code', ''),
                'confidence_score': float(optimization_data.get('confidence_score', 0.7)),
                'metadata': {
                    'original_component': analysis_data['component_name'],
                    'optimization_explanation': optimized_component.get('optimization_explanation', ''),
                    'breaking_changes': optimized_component.get('breaking_changes', []),
                    'migration_notes': optimized_component.get('migration_notes', ''),
                    'performance_expectations': optimized_component.get('performance_expectations', ''),
                    'analysis_results': optimization_data.get('analysis', {}),
                    'implementation_guide': optimization_data.get('implementation', {}),
                    'timestamp': datetime.now().isoformat()
                }
            }
        
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic optimization result
            return {
                'component_name': analysis_data['component_name'],
                'optimization_type': analysis_data.get('optimization_type', 'performance'),
                'optimized_code': content,
                'confidence_score': 0.5,
                'metadata': {
                    'raw_response': True,
                    'parsing_failed': True,
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def _format_clock_app_code(self, clock_app_data: Dict[str, Any]) -> str:
        """Format the generated clock app into organized code."""
        components = clock_app_data.get('components', {})
        deployment = clock_app_data.get('deployment', {})
        
        formatted_code = f'''"""
LLMFlow Clock Application
Generated by LLM Component Optimizer

A complete graph-based real-time clock application demonstrating LLMFlow's capabilities.
"""

# ========================================
# DEPLOYMENT INFORMATION
# ========================================

# Requirements: {', '.join(deployment.get('requirements', []))}

# Setup Instructions:
{chr(10).join(f"# {i+1}. {step}" for i, step in enumerate(deployment.get('setup_instructions', [])))}

# Run Instructions: {deployment.get('run_instructions', 'python main.py')}

# ========================================
# GENERATED COMPONENTS
# ========================================

'''
        
        # Add each component
        for category, items in components.items():
            formatted_code += f"\n# {category.upper()}\n"
            formatted_code += "# " + "="*50 + "\n\n"
            
            if isinstance(items, dict):
                for component_name, component_code in items.items():
                    formatted_code += f"# {component_name}\n"
                    formatted_code += component_code + "\n\n"
            else:
                formatted_code += str(items) + "\n\n"
        
        return formatted_code
    
    async def _write_clock_app_files(self, clock_app_data: Dict[str, Any]):
        """Write the generated clock app files to the filesystem."""
        try:
            # Create clock app directory
            clock_app_dir = Path("examples/clock_app")
            clock_app_dir.mkdir(parents=True, exist_ok=True)
            
            components = clock_app_data.get('components', {})
            
            # Write component files
            for category, items in components.items():
                if isinstance(items, dict):
                    for component_name, component_code in items.items():
                        file_path = clock_app_dir / f"{component_name.lower()}.py"
                        with open(file_path, 'w') as f:
                            f.write(f'"""\n{component_name}\nGenerated by LLMFlow LLM Optimizer\n"""\n\n')
                            f.write(component_code)
                        logger.info(f"Created {file_path}")
            
            # Write main app file
            if 'main_app' in components:
                main_file = clock_app_dir / "main.py"
                with open(main_file, 'w') as f:
                    f.write(components['main_app'])
                logger.info(f"Created {main_file}")
            
            # Write README
            readme_content = f"""# LLMFlow Clock Application

{clock_app_data.get('application', {}).get('description', 'Graph-based real-time clock application')}

## Architecture
{clock_app_data.get('application', {}).get('architecture', 'Distributed queue-based with data flow graph')}

## Setup
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(clock_app_data.get('deployment', {}).get('setup_instructions', [])))}

## Running
{clock_app_data.get('deployment', {}).get('run_instructions', 'python main.py')}

## Requirements
{chr(10).join(f'- {req}' for req in clock_app_data.get('deployment', {}).get('requirements', []))}

Generated by LLMFlow LLM Optimizer on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            readme_file = clock_app_dir / "README.md"
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            logger.info(f"Created {readme_file}")
            
        except Exception as e:
            logger.warning(f"Failed to write clock app files: {e}")
    
    def _create_error_component(self, error_message: str) -> OptimizedComponentAtom:
        """Create an error component."""
        error_component = {
            'component_name': 'error',
            'optimization_type': 'error',
            'optimized_code': '',
            'confidence_score': 0.0,
            'metadata': {
                'error': True,
                'error_message': error_message,
                'timestamp': datetime.now().isoformat()
            }
        }
        return OptimizedComponentAtom(error_component)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        avg_confidence = (
            self.stats['total_confidence_score'] / max(self.stats['successful_optimizations'], 1)
        )
        
        return {
            'optimizer': 'llm_component_optimizer',
            'stats': self.stats.copy(),
            'performance': {
                'success_rate': (
                    self.stats['successful_optimizations'] / 
                    max(self.stats['optimizations_performed'], 1)
                ),
                'average_confidence': avg_confidence,
                'optimization_types': self.stats['optimization_types'].copy()
            },
            'config': {
                'model': 'google/gemini-2.0-flash-001',
                'api_provider': 'openrouter',
                'max_tokens': 8000
            },
            'timestamp': datetime.now().isoformat()
        }


# Helper function to create a clock app generation request
def create_clock_app_request() -> ComponentAnalysisAtom:
    """Create a request to generate the complete clock application."""
    analysis_data = {
        'component_name': 'clock_app_generator',
        'current_code': '# Generate complete clock application',
        'performance_metrics': {
            'target': 'real_time_updates',
            'architecture': 'graph_based_data_flow',
            'communication': 'queue_only'
        },
        'analysis_timestamp': datetime.now().isoformat(),
        'optimization_type': 'complete_application_generation',
        'target_improvement': 100,
        'constraints': 'use LLMFlow architecture patterns'
    }
    
    return ComponentAnalysisAtom(analysis_data)
