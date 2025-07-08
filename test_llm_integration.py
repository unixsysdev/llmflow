#!/usr/bin/env python3
"""
LLM Integration Test

This script tests the new OpenAI-powered optimization system in LLMFlow.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMIntegrationTester:
    """Test LLM integration functionality."""
    
    def __init__(self):
        self.test_results = []
    
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name} - {details}")
        self.test_results.append((test_name, passed, details))
    
    async def test_openai_dependency(self):
        """Test OpenAI library installation."""
        try:
            import openai
            from openai import AsyncOpenAI
            self.log_result("OpenAI Dependency", True, f"OpenAI library available")
        except ImportError as e:
            self.log_result("OpenAI Dependency", False, f"OpenAI library not installed: {e}")
    
    async def test_llm_atoms_import(self):
        """Test LLM atoms can be imported."""
        try:
            from llmflow.atoms.llm import (
                LLMRequest, LLMResponse, LLMRequestAtom, LLMResponseAtom, OpenAIServiceAtom
            )
            self.log_result("LLM Atoms Import", True, "All LLM atoms imported successfully")
        except ImportError as e:
            self.log_result("LLM Atoms Import", False, f"Import failed: {e}")
    
    async def test_configuration_loading(self):
        """Test LLM configuration loading."""
        try:
            from llmflow.plugins.config import load_default_configuration
            
            config = load_default_configuration()
            
            if 'llm' in config:
                llm_config = config['llm']
                required_keys = ['provider', 'model', 'max_tokens', 'temperature']
                
                if all(key in llm_config for key in required_keys):
                    self.log_result("Configuration Loading", True, f"LLM config loaded: {llm_config['provider']}")
                else:
                    missing = [key for key in required_keys if key not in llm_config]
                    self.log_result("Configuration Loading", False, f"Missing keys: {missing}")
            else:
                self.log_result("Configuration Loading", False, "No LLM configuration found")
        
        except Exception as e:
            self.log_result("Configuration Loading", False, f"Config error: {e}")
    
    def print_results(self):
        """Print test results."""
        print("\n" + "="*80)
        print("🤖 LLM INTEGRATION TEST RESULTS")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        
        print(f"\n📊 SUMMARY: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        for test_name, passed, details in self.test_results:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}: {details}")
        
        print("\n" + "="*80)
        
        if passed_tests == total_tests:
            print("🎉 ALL LLM INTEGRATION TESTS PASSED!")
            print("✅ OpenAI-powered optimization is ready to use!")
            print("\n🔑 Next Steps:")
            print("1. Set OPENAI_API_KEY environment variable")
            print("2. Run the LLMFlow system with optimization enabled")
            print("3. Monitor optimization recommendations in the master queue")
        else:
            print(f"⚠️  {total_tests - passed_tests} tests failed - review issues above")
            print("\n🔧 Common Solutions:")
            print("- Install OpenAI: pip install openai>=1.0.0")
            print("- Check import paths and module structure")
            print("- Verify configuration loading")
        
        print("="*80)
        
        return passed_tests == total_tests

async def main():
    """Main test function."""
    print("🤖 LLMFlow OpenAI Integration Test")
    print("="*50)
    
    tester = LLMIntegrationTester()
    
    # Run basic tests
    await tester.test_openai_dependency()
    await tester.test_llm_atoms_import()
    await tester.test_configuration_loading()
    
    # Print results
    success = tester.print_results()
    
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
