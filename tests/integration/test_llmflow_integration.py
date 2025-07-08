#!/usr/bin/env python3
"""
Test LLMFlow Core Integration

This script tests the core LLMFlow components to ensure everything works together.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add llmflow to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core components can be imported."""
    print("🔍 Testing LLMFlow Core Component Imports")
    print("=" * 50)
    
    try:
        from llmflow.core.graph import GraphDefinition, create_clock_app_graph
        print("✅ Graph definition system")
        
        from llmflow.llm.component_generator import LLMComponentGenerator
        print("✅ LLM component generator")
        
        from llmflow.conductor.llm_conductor import LLMOptimizationConductor
        print("✅ LLM optimization conductor")
        
        from llmflow.queue.manager import QueueManager
        print("✅ Queue manager")
        
        from llmflow.atoms.openrouter_llm import OpenRouterServiceAtom
        print("✅ OpenRouter LLM service")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_graph_creation():
    """Test graph creation and validation."""
    print("\n🏗️ Testing Graph Creation and Validation")
    print("=" * 50)
    
    try:
        from llmflow.core.graph import create_clock_app_graph
        
        # Create graph
        graph = create_clock_app_graph()
        print(f"✅ Created graph: {graph.name}")
        print(f"   Components: {len(graph.components)}")
        print(f"   Connections: {len(graph.connections)}")
        
        # Validate graph
        validation = graph.validate()
        if validation.is_valid:
            print("✅ Graph validation passed")
        else:
            print(f"❌ Graph validation failed: {validation.errors}")
            return False
        
        # Test deployment order
        deployment_order = graph.get_deployment_order()
        print(f"✅ Deployment order calculated: {len(deployment_order)} components")
        
        # Test JSON export
        graph_json = graph.to_json()
        print(f"✅ JSON export: {len(graph_json)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph creation failed: {e}")
        return False

async def test_llm_service():
    """Test LLM service connectivity."""
    print("\n🤖 Testing LLM Service Connectivity")
    print("=" * 50)
    
    try:
        from llmflow.atoms.openrouter_llm import OpenRouterServiceAtom, OpenRouterRequest, OpenRouterRequestAtom
        
        # Check API key
        api_key = os.getenv('OPENROUTER_API_KEY') or "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"
        print(f"✅ API key configured: {api_key[:8]}...{api_key[-4:]}")
        
        # Create service
        llm_service = OpenRouterServiceAtom()
        print("✅ LLM service created")
        
        # Test simple request
        request = OpenRouterRequest(
            prompt="Say 'Hello from LLMFlow test!' in exactly those words.",
            model="google/gemini-2.0-flash-001",
            max_tokens=50,
            temperature=0.0
        )
        
        print("🔄 Testing LLM connectivity...")
        response_atoms = await llm_service.process([OpenRouterRequestAtom(request)])
        
        if response_atoms and not response_atoms[0].response.error:
            response = response_atoms[0].response
            print(f"✅ LLM response received:")
            print(f"   Content: {response.content}")
            print(f"   Model: {response.model}")
            print(f"   Cost: ${response.cost_usd:.4f}")
            print(f"   Confidence: {response.confidence_score:.2f}")
            return True
        else:
            error = response_atoms[0].response.error if response_atoms else "No response"
            print(f"❌ LLM request failed: {error}")
            return False
        
    except Exception as e:
        print(f"❌ LLM service test failed: {e}")
        return False

async def test_component_generation():
    """Test component generation."""
    print("\n⚙️ Testing Component Generation")
    print("=" * 50)
    
    try:
        from llmflow.core.graph import create_clock_app_graph
        from llmflow.llm.component_generator import LLMComponentGenerator
        from llmflow.core.base import ComponentType
        
        # Create graph
        graph = create_clock_app_graph()
        print(f"✅ Graph loaded: {graph.name}")
        
        # Create generator
        generator = LLMComponentGenerator()
        print("✅ Component generator created")
        
        # Test single component generation
        atoms = graph.get_components_by_type(ComponentType.ATOM)
        if atoms:
            test_component = atoms[0]
            print(f"🔄 Testing generation of: {test_component.name}")
            
            generated = await generator._generate_single_component(test_component, graph, {})
            
            if generated:
                print(f"✅ Component generated successfully!")
                print(f"   Confidence: {generated.confidence:.1%}")
                print(f"   Code size: {len(generated.generated_code)} characters")
                print(f"   Preview: {generated.generated_code[:100]}...")
                return True
            else:
                print("❌ Component generation failed")
                return False
        else:
            print("❌ No atoms found in graph")
            return False
        
    except Exception as e:
        print(f"❌ Component generation test failed: {e}")
        return False

def test_conductor_creation():
    """Test conductor creation."""
    print("\n🎛️ Testing Conductor Creation")
    print("=" * 50)
    
    try:
        from llmflow.conductor.llm_conductor import LLMOptimizationConductor
        from llmflow.queue.manager import QueueManager
        
        # Create queue manager
        queue_manager = QueueManager()
        print("✅ Queue manager created")
        
        # Create conductor
        conductor = LLMOptimizationConductor(queue_manager)
        print("✅ LLM optimization conductor created")
        
        # Check configuration
        status = conductor.get_llm_optimization_status()
        print(f"✅ Conductor configuration:")
        print(f"   LLM enabled: {status['llm_optimization_enabled']}")
        print(f"   Model: {status['llm_model']}")
        print(f"   API key: {'configured' if status['api_key_configured'] else 'missing'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conductor creation failed: {e}")
        return False

async def run_all_tests():
    """Run all integration tests."""
    print("🧪 LLMFlow Core Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Graph Creation", test_graph_creation),
        ("LLM Service", test_llm_service),
        ("Component Generation", test_component_generation),
        ("Conductor Creation", test_conductor_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append((test_name, result))
            
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - LLMFlow core integration is working!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed - check output above")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)
