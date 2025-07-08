#!/usr/bin/env python3
"""
LLMFlow LLM Optimization Demo

This script demonstrates the LLM-powered optimization capabilities of LLMFlow,
including the automatic generation of a complete graph-based clock application.
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

# Add the llmflow package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from llmflow.atoms.llm_optimizer import (
        LLMComponentOptimizerAtom, 
        ComponentAnalysisAtom,
        create_clock_app_request
    )
    from llmflow.atoms.openrouter_llm import OpenRouterServiceAtom
    from llmflow.queue.manager import QueueManager
    LLMFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LLMFlow components not available: {e}")
    print("   This is normal for the demo - running in standalone mode")
    LLMFLOW_AVAILABLE = False
    
    # Mock classes for demo
    class MockQueueManager:
        pass
    class MockOptimizerAtom:
        def __init__(self, *args):
            pass
        async def process(self, inputs):
            return [MockResult()]
        def get_stats(self):
            return {"demo": "mock_stats"}
    class MockResult:
        def __init__(self):
            self.component_data = {
                'component_name': 'demo_component',
                'optimized_code': 'print("Demo generated code")',
                'confidence_score': 0.95,
                'metadata': {'demo': True}
            }
    
    QueueManager = MockQueueManager
    LLMComponentOptimizerAtom = MockOptimizerAtom
    OpenRouterServiceAtom = MockOptimizerAtom

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_component_optimization():
    """Demonstrate component optimization."""
    print("🚀 LLMFlow LLM Optimization Demo")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("⚠️  Setting OpenRouter API key from demo...")
        os.environ['OPENROUTER_API_KEY'] = "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"
    
    # Initialize services
    print("🔧 Initializing LLM optimization services...")
    queue_manager = QueueManager()
    llm_optimizer = LLMComponentOptimizerAtom(queue_manager)
    
    # Test OpenRouter connection
    print("🌐 Testing OpenRouter connection...")
    openrouter_service = OpenRouterServiceAtom()
    
    # Demo 1: Generate complete clock application
    print("\n📱 Demo 1: Generating Complete Clock Application")
    print("-" * 50)
    
    try:
        # Create clock app generation request
        clock_request = create_clock_app_request()
        print("✅ Created clock app generation request")
        
        # Process the request
        print("🤖 Sending request to LLM for clock app generation...")
        print("   This may take 30-60 seconds...")
        
        optimized_components = await llm_optimizer.process([clock_request])
        
        if optimized_components and not optimized_components[0].component_data.get('metadata', {}).get('error'):
            clock_component = optimized_components[0]
            
            print("✅ Clock application generated successfully!")
            print(f"   Confidence Score: {clock_component.component_data['confidence_score']:.2f}")
            print(f"   Components Generated: {len(clock_component.component_data.get('app_structure', {}).get('components', {}))}")
            
            # Show some of the generated code
            code_preview = clock_component.component_data['optimized_code'][:500]
            print(f"\n📝 Code Preview (first 500 chars):")
            print("-" * 30)
            print(code_preview + "...")
            
            # Save the complete application
            output_file = Path("generated_clock_app.py")
            with open(output_file, 'w') as f:
                f.write(clock_component.component_data['optimized_code'])
            print(f"\n💾 Complete application saved to: {output_file}")
            
            # Show deployment info if available
            deployment_info = clock_component.component_data.get('deployment_info', {})
            if deployment_info:
                print(f"\n🚀 Deployment Instructions:")
                for i, step in enumerate(deployment_info.get('setup_instructions', []), 1):
                    print(f"   {i}. {step}")
                print(f"   Run: {deployment_info.get('run_instructions', 'python main.py')}")
        
        else:
            error_msg = optimized_components[0].component_data.get('metadata', {}).get('error_message', 'Unknown error')
            print(f"❌ Clock app generation failed: {error_msg}")
    
    except Exception as e:
        print(f"❌ Clock app generation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Demo 2: Optimize a sample component
    print("\n⚡ Demo 2: Component Optimization")
    print("-" * 50)
    
    try:
        # Create a sample component for optimization
        sample_component_code = '''
def slow_fibonacci(n):
    """Inefficient fibonacci implementation."""
    if n <= 1:
        return n
    return slow_fibonacci(n-1) + slow_fibonacci(n-2)

class DataProcessor:
    """Sample data processor with performance issues."""
    
    def __init__(self):
        self.data = []
    
    def add_data(self, item):
        # Inefficient - creates new list each time
        self.data = self.data + [item]
    
    def process_data(self):
        # Inefficient - multiple loops
        result = []
        for item in self.data:
            if item > 0:
                for i in range(len(self.data)):
                    if self.data[i] == item:
                        result.append(item * 2)
                        break
        return result
'''
        
        # Create optimization request
        optimization_request = ComponentAnalysisAtom({
            'component_name': 'sample_data_processor',
            'current_code': sample_component_code,
            'performance_metrics': {
                'avg_execution_time': 2.5,
                'memory_usage': '150MB',
                'cpu_utilization': '85%',
                'bottlenecks': ['recursive calls', 'list concatenation', 'nested loops']
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'optimization_type': 'performance',
            'target_improvement': 50,
            'constraints': 'maintain API compatibility'
        })
        
        print("✅ Created component optimization request")
        print("🤖 Sending request to LLM for optimization...")
        print("   This may take 30-45 seconds...")
        
        # Process optimization
        optimized_components = await llm_optimizer.process([optimization_request])
        
        if optimized_components and not optimized_components[0].component_data.get('metadata', {}).get('error'):
            optimized_component = optimized_components[0]
            
            print("✅ Component optimized successfully!")
            print(f"   Confidence Score: {optimized_component.component_data['confidence_score']:.2f}")
            
            # Show optimization details
            metadata = optimized_component.component_data.get('metadata', {})
            if 'optimization_explanation' in metadata:
                print(f"\n📈 Optimization Summary:")
                print(f"   {metadata['optimization_explanation'][:200]}...")
            
            # Save optimized code
            optimized_output = Path("optimized_component.py")
            with open(optimized_output, 'w') as f:
                f.write(optimized_component.component_data['optimized_code'])
            print(f"\n💾 Optimized code saved to: {optimized_output}")
        
        else:
            error_msg = optimized_components[0].component_data.get('metadata', {}).get('error_message', 'Unknown error')
            print(f"❌ Component optimization failed: {error_msg}")
    
    except Exception as e:
        print(f"❌ Component optimization error: {e}")
        import traceback
        traceback.print_exc()
    
    # Show statistics
    print("\n📊 LLM Optimizer Statistics")
    print("-" * 50)
    stats = llm_optimizer.get_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n🎉 Demo completed!")
    print("💡 Check the generated files:")
    print("   - generated_clock_app.py: Complete clock application")
    print("   - optimized_component.py: Optimized sample component")
    print("   - examples/clock_app/: Individual component files (if generated)")


async def demo_openrouter_features():
    """Demonstrate OpenRouter service features."""
    print("\n🌟 OpenRouter Service Features Demo")
    print("-" * 50)
    
    try:
        from llmflow.atoms.openrouter_llm import OpenRouterRequestAtom, OpenRouterRequest
        
        openrouter_service = OpenRouterServiceAtom()
        
        # Test basic request
        request = OpenRouterRequest(
            prompt="Explain the benefits of graph-based data flow architecture in 100 words.",
            model="google/gemini-2.0-flash-001",
            site_url="https://llmflow.dev",
            site_name="LLMFlow Demo"
        )
        
        print("🤖 Testing OpenRouter with Gemini 2.0 Flash...")
        response_atoms = await openrouter_service.process([OpenRouterRequestAtom(request)])
        
        if response_atoms and not response_atoms[0].response.error:
            response = response_atoms[0].response
            print("✅ OpenRouter request successful!")
            print(f"   Model: {response.model}")
            print(f"   Tokens: {response.usage_tokens}")
            print(f"   Cost: ${response.cost_usd:.4f}")
            print(f"   Confidence: {response.confidence_score:.2f}")
            print(f"\n📝 Response:")
            print(f"   {response.content[:200]}...")
        else:
            error = response_atoms[0].response.error if response_atoms else "No response"
            print(f"❌ OpenRouter request failed: {error}")
        
        # Show service stats
        print("\n📊 OpenRouter Service Stats:")
        stats = openrouter_service.get_stats()
        print(json.dumps({
            'total_requests': stats['stats']['total_requests'],
            'success_rate': stats['stats']['successful_requests'] / max(stats['stats']['total_requests'], 1),
            'total_cost': f"${stats['stats']['total_cost_usd']:.4f}",
            'supported_models': len(stats['config']['supported_models'])
        }, indent=2))
    
    except Exception as e:
        print(f"❌ OpenRouter demo error: {e}")


if __name__ == "__main__":
    print("🎯 Starting LLMFlow LLM Optimization Demo")
    
    try:
        # Run the main demo
        asyncio.run(demo_component_optimization())
        
        # Run OpenRouter features demo
        asyncio.run(demo_openrouter_features())
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Demo finished!")
