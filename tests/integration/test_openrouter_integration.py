#!/usr/bin/env python3
"""
Test OpenRouter LLM Integration

This script tests the OpenRouter integration and LLM optimization capabilities.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Test 1: Check if OpenAI library is available
print("🔍 Testing OpenRouter Integration")
print("=" * 50)

try:
    from openai import AsyncOpenAI
    print("✅ OpenAI library available")
    openai_available = True
except ImportError:
    print("❌ OpenAI library not available. Install with: pip install openai>=1.0.0")
    openai_available = False

# Test 2: Check OpenRouter connection
if openai_available:
    async def test_openrouter_connection():
        """Test basic OpenRouter connection."""
        api_key = os.getenv('OPENROUTER_API_KEY') or "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"
        
        try:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            
            print("🌐 Testing OpenRouter API connection...")
            
            response = await client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": "Say 'Hello from LLMFlow!' in exactly those words."}],
                max_tokens=50,
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            
            print(f"✅ OpenRouter connection successful!")
            print(f"   Model: google/gemini-2.0-flash-001")
            print(f"   Response: {content}")
            print(f"   Tokens used: {tokens}")
            
            return True
            
        except Exception as e:
            print(f"❌ OpenRouter connection failed: {e}")
            return False
    
    # Run connection test
    try:
        connection_success = asyncio.run(test_openrouter_connection())
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        connection_success = False
else:
    connection_success = False

# Test 3: Test LLM optimization components (if available)
print(f"\n🤖 Testing LLM Optimization Components")
print("-" * 50)

try:
    # Try to import our new components
    sys.path.insert(0, str(Path(__file__).parent))
    
    from llmflow.atoms.openrouter_llm import (
        OpenRouterServiceAtom, 
        OpenRouterRequest, 
        OpenRouterRequestAtom
    )
    print("✅ OpenRouter atoms import successfully")
    
    from llmflow.atoms.llm_optimizer import (
        LLMComponentOptimizerAtom,
        ComponentAnalysisAtom,
        create_clock_app_request
    )
    print("✅ LLM optimizer atoms import successfully")
    
    components_available = True

except ImportError as e:
    print(f"❌ Component import failed: {e}")
    print("   This is expected if LLMFlow dependencies are not installed")
    components_available = False

# Test 4: Demo request creation
if components_available and openai_available:
    print(f"\n⚡ Testing LLM Request Creation")
    print("-" * 50)
    
    try:
        # Test OpenRouter request creation
        request = OpenRouterRequest(
            prompt="Generate a simple 'Hello World' function in Python.",
            model="google/gemini-2.0-flash-001",
            site_url="https://llmflow.dev",
            site_name="LLMFlow Test"
        )
        
        request_atom = OpenRouterRequestAtom(request)
        validation = request_atom.validate()
        
        if validation.is_valid:
            print("✅ OpenRouter request creation and validation works")
        else:
            print(f"❌ Request validation failed: {validation.errors}")
        
        # Test clock app request creation
        clock_request = create_clock_app_request()
        clock_validation = clock_request.validate()
        
        if clock_validation.is_valid:
            print("✅ Clock app request creation works")
        else:
            print(f"❌ Clock app request validation failed: {clock_validation.errors}")
            
    except Exception as e:
        print(f"❌ Request creation test failed: {e}")

# Test 5: Simple optimization demo (if everything works)
if components_available and openai_available and connection_success:
    print(f"\n🚀 Running Simple LLM Optimization Demo")
    print("-" * 50)
    
    async def simple_optimization_demo():
        """Run a simple optimization demo."""
        try:
            # Create service
            openrouter_service = OpenRouterServiceAtom()
            
            # Create simple request
            request = OpenRouterRequest(
                prompt="Optimize this Python function for better performance:\n\ndef slow_sum(numbers):\n    total = 0\n    for i in range(len(numbers)):\n        total = total + numbers[i]\n    return total\n\nProvide just the optimized function code.",
                model="google/gemini-2.0-flash-001",
                max_tokens=200,
                temperature=0.1
            )
            
            print("🤖 Asking Gemini 2.0 Flash to optimize a simple function...")
            
            # Process request
            result = await openrouter_service.process([OpenRouterRequestAtom(request)])
            
            if result and not result[0].response.error:
                response = result[0].response
                print("✅ LLM optimization successful!")
                print(f"   Model: {response.model}")
                print(f"   Tokens: {response.usage_tokens}")
                print(f"   Cost: ${response.cost_usd:.4f}")
                print(f"   Confidence: {response.confidence_score:.2f}")
                print(f"\n💻 Generated optimization:")
                print(f"   {response.content[:200]}...")
                
                # Show stats
                stats = openrouter_service.get_stats()
                print(f"\n📊 Service Stats:")
                print(f"   Requests: {stats['stats']['total_requests']}")
                print(f"   Success Rate: {stats['stats']['successful_requests']}/{stats['stats']['total_requests']}")
                print(f"   Total Cost: ${stats['stats']['total_cost_usd']:.4f}")
                
                return True
            else:
                error = result[0].response.error if result else "No response"
                print(f"❌ LLM optimization failed: {error}")
                return False
                
        except Exception as e:
            print(f"❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    try:
        demo_success = asyncio.run(simple_optimization_demo())
    except Exception as e:
        print(f"❌ Demo execution error: {e}")
        demo_success = False
else:
    demo_success = False

# Final report
print(f"\n🎯 Test Results Summary")
print("=" * 50)
print(f"OpenAI Library:      {'✅' if openai_available else '❌'}")
print(f"OpenRouter API:      {'✅' if connection_success else '❌'}")
print(f"LLM Components:      {'✅' if components_available else '❌'}")
print(f"Optimization Demo:   {'✅' if demo_success else '❌'}")

if all([openai_available, connection_success, components_available, demo_success]):
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"✅ OpenRouter LLM integration is fully working!")
    print(f"🚀 Ready to run: python demo_llm_optimization.py")
elif openai_available and connection_success:
    print(f"\n⚠️  OpenRouter API works, but LLMFlow components need setup")
    print(f"💡 Try: pip install -r requirements.txt")
else:
    print(f"\n❌ Setup required:")
    if not openai_available:
        print(f"   pip install openai>=1.0.0")
    if not connection_success:
        print(f"   Check OpenRouter API key and internet connection")
    
print(f"\n👋 Test completed!")
