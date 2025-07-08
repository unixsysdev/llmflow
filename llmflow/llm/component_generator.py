"""
LLM Component Generator

This module uses Gemini 2.0 Flash to generate actual working Python code from
graph definitions. It takes a GraphDefinition and produces complete, functional
LLMFlow components that follow the framework patterns.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from ..core.graph import GraphDefinition, ComponentSpec, ConnectionType
from ..core.base import ComponentType
from ..atoms.openrouter_llm import OpenRouterServiceAtom, OpenRouterRequest, OpenRouterRequestAtom

logger = logging.getLogger(__name__)


class GeneratedComponent:
    """Container for a generated component with its code and metadata."""
    
    def __init__(self, component_spec: ComponentSpec, generated_code: str, 
                 confidence: float, metadata: Dict[str, Any] = None):
        self.component_spec = component_spec
        self.generated_code = generated_code
        self.confidence = confidence
        self.metadata = metadata or {}
        self.generated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component_spec': {
                'id': self.component_spec.id,
                'name': self.component_spec.name,
                'component_type': self.component_spec.component_type.value,
                'description': self.component_spec.description
            },
            'generated_code': self.generated_code,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'generated_at': self.generated_at.isoformat()
        }


class LLMComponentGenerator:
    """Generates working LLMFlow components from graph definitions using Gemini 2.0 Flash."""
    
    def __init__(self):
        # Set up OpenRouter API key
        self.api_key = (
            os.getenv('OPENROUTER_API_KEY') or 
            "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"
        )
        os.environ['OPENROUTER_API_KEY'] = self.api_key
        
        # Initialize LLM service
        self.llm_service = OpenRouterServiceAtom()
        
        # Generation configuration
        self.config = {
            'model': 'google/gemini-2.0-flash-001',
            'max_tokens': 8000,
            'temperature': 0.1,
            'confidence_threshold': 0.8,
            'max_retries': 3
        }
        
        # Statistics
        self.stats = {
            'components_generated': 0,
            'total_cost_usd': 0.0,
            'average_confidence': 0.0,
            'generation_failures': 0
        }
        
        logger.info("LLM Component Generator initialized with Gemini 2.0 Flash")
    
    async def generate_application_from_graph(self, graph: GraphDefinition) -> Dict[str, GeneratedComponent]:
        """Generate complete application from graph definition."""
        logger.info(f"🏗️ Generating application: {graph.name}")
        
        # Validate graph first
        validation = graph.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid graph: {validation.errors}")
        
        generated_components = {}
        
        # Get deployment order to generate dependencies first
        deployment_order = graph.get_deployment_order()
        
        # Generate components in dependency order
        for component_id in deployment_order:
            component_spec = graph.get_component(component_id)
            if not component_spec:
                continue
            
            logger.info(f"🔧 Generating component: {component_spec.name} ({component_spec.component_type.value})")
            
            try:
                generated_component = await self._generate_single_component(
                    component_spec, graph, generated_components
                )
                
                if generated_component:
                    generated_components[component_id] = generated_component
                    self.stats['components_generated'] += 1
                    logger.info(f"✅ Generated {component_spec.name} (confidence: {generated_component.confidence:.1%})")
                else:
                    logger.error(f"❌ Failed to generate {component_spec.name}")
                    self.stats['generation_failures'] += 1
            
            except Exception as e:
                logger.error(f"❌ Error generating {component_spec.name}: {e}")
                self.stats['generation_failures'] += 1
        
        # Calculate average confidence
        if generated_components:
            total_confidence = sum(comp.confidence for comp in generated_components.values())
            self.stats['average_confidence'] = total_confidence / len(generated_components)
        
        logger.info(f"🎉 Generated {len(generated_components)} components with {self.stats['average_confidence']:.1%} average confidence")
        
        return generated_components
    
    async def _generate_single_component(self, component_spec: ComponentSpec, 
                                       graph: GraphDefinition,
                                       existing_components: Dict[str, GeneratedComponent]) -> Optional[GeneratedComponent]:
        """Generate a single component."""
        
        # Create generation prompt based on component type
        prompt = self._create_generation_prompt(component_spec, graph, existing_components)
        
        # Send to LLM
        for attempt in range(self.config['max_retries']):
            try:
                response = await self._send_generation_request(prompt, component_spec)
                
                if response:
                    # Parse and validate the generated code
                    generated_component = await self._parse_generation_response(
                        response, component_spec
                    )
                    
                    if generated_component and generated_component.confidence >= self.config['confidence_threshold']:
                        return generated_component
                    elif generated_component:
                        logger.warning(f"Low confidence for {component_spec.name}: {generated_component.confidence:.1%}")
                        return generated_component  # Still return it but with warning
                
                logger.warning(f"Generation attempt {attempt + 1} failed for {component_spec.name}")
                
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} error for {component_spec.name}: {e}")
        
        return None
    
    def _create_generation_prompt(self, component_spec: ComponentSpec, 
                                graph: GraphDefinition,
                                existing_components: Dict[str, GeneratedComponent]) -> str:
        """Create LLM prompt for component generation."""
        
        # Get component connections
        input_connections = [
            conn for conn in graph.connections.values() 
            if conn.target_component == component_spec.id
        ]
        output_connections = [
            conn for conn in graph.connections.values()
            if conn.source_component == component_spec.id
        ]
        
        # Get dependency code context
        dependency_context = self._build_dependency_context(component_spec, existing_components)
        
        prompt = f"""
Generate a complete, working LLMFlow {component_spec.component_type.value} component.

## Component Specification
**Name**: {component_spec.name}
**Type**: {component_spec.component_type.value}
**Description**: {component_spec.description}

**Input Types**: {component_spec.input_types}
**Output Types**: {component_spec.output_types}
**Input Queues**: {component_spec.input_queues}
**Output Queues**: {component_spec.output_queues}

**Performance Requirements**:
- Max Latency: {component_spec.max_latency_ms}ms
- Max Memory: {component_spec.max_memory_mb}MB
- Max CPU: {component_spec.max_cpu_percent}%

**Implementation Hints**: {json.dumps(component_spec.implementation_hints, indent=2)}

## Graph Context
**Application**: {graph.name} - {graph.description}
**Performance Goals**: {json.dumps(graph.performance_goals, indent=2)}
**Global Config**: {json.dumps(graph.global_config, indent=2)}

## Connections
**Input Connections**:
{chr(10).join(f'  - {conn.source_component}.{conn.source_queue} → {conn.target_queue} ({conn.data_types})' for conn in input_connections)}

**Output Connections**:
{chr(10).join(f'  - {conn.source_queue} → {conn.target_component}.{conn.target_queue} ({conn.data_types})' for conn in output_connections)}

## Dependencies Context
{dependency_context}

## LLMFlow Framework Patterns

### For DataAtom (if component_type is ATOM and handles data):
```python
from llmflow.core.base import DataAtom, ValidationResult
import json
from datetime import datetime

class {component_spec.name}(DataAtom):
    \"\"\"Generated data atom for {component_spec.description}\"\"\"
    
    def __init__(self, value):
        super().__init__(value)
    
    def validate(self) -> ValidationResult:
        errors = []
        # Add validation logic based on component spec
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def serialize(self) -> bytes:
        return json.dumps(self.value).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes):
        return cls(json.loads(data.decode('utf-8')))
```

### For ServiceAtom (if component_type is ATOM and handles processing):
```python
from llmflow.core.base import ServiceAtom, DataAtom
import asyncio
from typing import List

class {component_spec.name}(ServiceAtom):
    \"\"\"Generated service atom for {component_spec.description}\"\"\"
    
    def __init__(self):
        super().__init__(
            name="{component_spec.name.lower()}",
            input_types={component_spec.input_types},
            output_types={component_spec.output_types}
        )
    
    async def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        # Implement processing logic based on component spec
        pass
```

### For Molecule (if component_type is MOLECULE):
```python
from llmflow.core.base import ServiceAtom, DataAtom
from llmflow.queue.manager import QueueManager
import asyncio
from typing import List, Dict, Any

class {component_spec.name}(ServiceAtom):
    \"\"\"Generated molecule for {component_spec.description}\"\"\"
    
    def __init__(self, queue_manager: QueueManager):
        super().__init__(
            name="{component_spec.name.lower()}",
            input_types={component_spec.input_types},
            output_types={component_spec.output_types}
        )
        self.queue_manager = queue_manager
        
        # Initialize any dependent atoms/services
        
    async def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        # Implement business logic based on component spec
        # Use queue communication for complex workflows
        pass
    
    async def start(self):
        \"\"\"Start the molecule and its components.\"\"\"
        pass
    
    async def stop(self):
        \"\"\"Stop the molecule and clean up.\"\"\"
        pass
```

### For Cell (if component_type is CELL):
```python
from llmflow.core.base import ServiceAtom, DataAtom
from llmflow.queue.manager import QueueManager
from llmflow.conductor.manager import ConductorManager
import asyncio
from typing import List, Dict, Any

class {component_spec.name}(ServiceAtom):
    \"\"\"Generated cell for {component_spec.description}\"\"\"
    
    def __init__(self, queue_manager: QueueManager, conductor_manager: ConductorManager):
        super().__init__(
            name="{component_spec.name.lower()}",
            input_types={component_spec.input_types},
            output_types={component_spec.output_types}
        )
        self.queue_manager = queue_manager
        self.conductor_manager = conductor_manager
        
        # Initialize molecules and orchestration
        
    async def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        # Implement application-level orchestration
        pass
    
    async def start(self):
        \"\"\"Start the application cell.\"\"\"
        pass
    
    async def stop(self):
        \"\"\"Stop the application cell.\"\"\"
        pass
```

## Requirements
1. **Generate complete, working Python code** that follows LLMFlow patterns
2. **Use queue-based communication** for all component interactions
3. **Include proper error handling** and logging
4. **Follow the component specification** exactly
5. **Implement the described functionality** based on hints and connections
6. **Use async/await** for all I/O operations
7. **Include proper imports** and dependencies
8. **Add comprehensive docstrings** and comments
9. **Optimize for the performance requirements** specified
10. **Make it production-ready** with proper validation

## Response Format (JSON):
{{
  "generated_code": "complete Python code for the component",
  "confidence_score": 0.0-1.0,
  "implementation_notes": "explanation of the implementation approach",
  "performance_optimizations": ["list of performance optimizations applied"],
  "dependencies": ["list of imported modules/components"],
  "testing_suggestions": ["suggestions for testing this component"],
  "deployment_notes": "notes for deploying this component"
}}

Generate production-ready, efficient, and well-documented code that perfectly implements the component specification.
"""
        
        return prompt
    
    def _build_dependency_context(self, component_spec: ComponentSpec, 
                                existing_components: Dict[str, GeneratedComponent]) -> str:
        """Build context about already generated dependencies."""
        context_parts = []
        
        for dep_id in component_spec.dependencies:
            if dep_id in existing_components:
                dep_component = existing_components[dep_id]
                context_parts.append(f"""
### Dependency: {dep_component.component_spec.name}
```python
# Already generated - reference this component
{dep_component.generated_code[:500]}...
```
""")
        
        if not context_parts:
            context_parts.append("No dependencies generated yet - this component should be self-contained.")
        
        return "\n".join(context_parts)
    
    async def _send_generation_request(self, prompt: str, component_spec: ComponentSpec) -> Optional[str]:
        """Send generation request to LLM."""
        try:
            request = OpenRouterRequest(
                prompt=prompt,
                model=self.config['model'],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                site_url="https://llmflow.dev",
                site_name="LLMFlow Component Generator"
            )
            
            response_atoms = await self.llm_service.process([OpenRouterRequestAtom(request)])
            
            if not response_atoms or response_atoms[0].response.error:
                error = response_atoms[0].response.error if response_atoms else "No response"
                logger.error(f"LLM request failed for {component_spec.name}: {error}")
                return None
            
            response = response_atoms[0].response
            self.stats['total_cost_usd'] += response.cost_usd
            
            logger.info(f"✅ LLM response for {component_spec.name} (${response.cost_usd:.4f})")
            
            return response.content
        
        except Exception as e:
            logger.error(f"Error sending LLM request for {component_spec.name}: {e}")
            return None
    
    async def _parse_generation_response(self, response_content: str, 
                                       component_spec: ComponentSpec) -> Optional[GeneratedComponent]:
        """Parse LLM response into GeneratedComponent."""
        try:
            # Try to parse as JSON first
            response_data = json.loads(response_content)
            
            generated_code = response_data.get('generated_code', '')
            confidence = float(response_data.get('confidence_score', 0.0))
            
            if not generated_code:
                logger.error(f"No generated code in response for {component_spec.name}")
                return None
            
            # Basic syntax validation
            try:
                compile(generated_code, f"{component_spec.name}.py", 'exec')
            except SyntaxError as e:
                logger.error(f"Generated code has syntax errors for {component_spec.name}: {e}")
                # Still return it but with lower confidence
                confidence *= 0.5
            
            metadata = {
                'implementation_notes': response_data.get('implementation_notes', ''),
                'performance_optimizations': response_data.get('performance_optimizations', []),
                'dependencies': response_data.get('dependencies', []),
                'testing_suggestions': response_data.get('testing_suggestions', []),
                'deployment_notes': response_data.get('deployment_notes', ''),
                'raw_response_size': len(response_content)
            }
            
            return GeneratedComponent(
                component_spec=component_spec,
                generated_code=generated_code,
                confidence=confidence,
                metadata=metadata
            )
        
        except json.JSONDecodeError:
            logger.warning(f"LLM response not in JSON format for {component_spec.name}")
            # Try to extract code from markdown blocks
            if '```python' in response_content:
                start = response_content.find('```python') + 9
                end = response_content.find('```', start)
                if end > start:
                    generated_code = response_content[start:end].strip()
                    
                    # Simple validation
                    if len(generated_code) > 50 and ('class' in generated_code or 'def' in generated_code):
                        return GeneratedComponent(
                            component_spec=component_spec,
                            generated_code=generated_code,
                            confidence=0.7,
                            metadata={'extracted_from_markdown': True}
                        )
            
            # Try to extract any code-like content
            if 'class' in response_content and 'def' in response_content:
                # Simple extraction - just return the response as-is if it contains code
                return GeneratedComponent(
                    component_spec=component_spec,
                    generated_code=response_content,
                    confidence=0.5,
                    metadata={'raw_response': True}
                )
            
            logger.error(f"Could not extract usable code from response for {component_spec.name}")
            return None
        except Exception as e:
            logger.error(f"Error parsing generation response for {component_spec.name}: {e}")
            return None
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            'generator': 'llm_component_generator',
            'model': self.config['model'],
            'stats': self.stats.copy(),
            'config': self.config.copy()
        }


async def demo_component_generation():
    """Demo the component generation system."""
    print("🤖 LLM Component Generator Demo")
    print("=" * 50)
    
    # Import the graph definition
    from .graph import create_clock_app_graph
    
    # Create graph
    graph = create_clock_app_graph()
    print(f"📱 Loaded graph: {graph.name}")
    print(f"   Components to generate: {len(graph.components)}")
    
    # Initialize generator
    generator = LLMComponentGenerator()
    
    # Generate all components
    print(f"\n🏗️ Generating components with Gemini 2.0 Flash...")
    generated_components = await generator.generate_application_from_graph(graph)
    
    print(f"\n✅ Generation complete!")
    print(f"   Generated: {len(generated_components)} components")
    
    # Show results
    for comp_id, generated_comp in generated_components.items():
        spec = generated_comp.component_spec
        print(f"\n📦 {spec.name} ({spec.component_type.value})")
        print(f"   Confidence: {generated_comp.confidence:.1%}")
        print(f"   Code size: {len(generated_comp.generated_code)} characters")
        print(f"   Preview: {generated_comp.generated_code[:100]}...")
    
    # Save generated components
    output_dir = Path("generated_components")
    output_dir.mkdir(exist_ok=True)
    
    for comp_id, generated_comp in generated_components.items():
        spec = generated_comp.component_spec
        
        # Save code file
        code_file = output_dir / f"{spec.name.lower()}.py"
        with open(code_file, 'w') as f:
            f.write(f'"""\n{spec.name}\nGenerated by LLM Component Generator\nConfidence: {generated_comp.confidence:.1%}\n"""\n\n')
            f.write(generated_comp.generated_code)
        
        # Save metadata
        meta_file = output_dir / f"{spec.name.lower()}_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(generated_comp.to_dict(), f, indent=2)
        
        print(f"💾 Saved {spec.name} to {code_file}")
    
    # Show statistics
    stats = generator.get_generation_stats()
    print(f"\n📊 Generation Statistics:")
    print(json.dumps(stats, indent=2))
    
    print(f"\n🎉 Component generation demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_component_generation())
