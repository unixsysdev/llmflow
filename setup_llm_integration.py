#!/usr/bin/env python3
"""
LLMFlow OpenAI Setup Script

Quick setup script for enabling OpenAI-powered optimization in LLMFlow.
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required dependencies are installed."""
    print("🔍 Checking requirements...")
    
    missing_deps = []
    
    try:
        import openai
        print("✅ OpenAI library installed")
    except ImportError:
        missing_deps.append("openai>=1.0.0")
        print("❌ OpenAI library not installed")
    
    try:
        import aiohttp
        print("✅ aiohttp library installed")
    except ImportError:
        missing_deps.append("aiohttp")
        print("❌ aiohttp library not installed")
    
    if missing_deps:
        print(f"\n📦 Install missing dependencies:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True

def check_api_key():
    """Check if OpenAI API key is configured."""
    print("\n🔑 Checking OpenAI API key...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✅ API key found: {masked_key}")
        return True
    else:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("\n🛠️  Setup instructions:")
        print("1. Get your API key from: https://platform.openai.com/api-keys")
        print("2. Set environment variable:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("3. Or create .env file with OPENAI_API_KEY=your-api-key-here")
        return False

def test_llm_integration():
    """Test LLM integration."""
    print("\n🧪 Testing LLM integration...")
    
    try:
        # Test configuration loading
        from llmflow.plugins.config import load_default_configuration
        config = load_default_configuration()
        
        if 'llm' in config:
            print("✅ LLM configuration loaded")
        else:
            print("❌ LLM configuration missing")
            return False
        
        # Test LLM atoms import
        from llmflow.atoms.llm import OpenAIServiceAtom, LLMRequest, LLMRequestAtom
        print("✅ LLM atoms imported")
        
        # Test service initialization
        service = OpenAIServiceAtom()
        print("✅ OpenAI service initialized")
        
        # Test request creation
        request = LLMRequest(
            prompt="Test prompt",
            model="gpt-4",
            temperature=0.1
        )
        request_atom = LLMRequestAtom(request)
        validation = request_atom.validate()
        
        if validation.is_valid:
            print("✅ LLM request validation works")
        else:
            print(f"❌ LLM request validation failed: {validation.errors}")
            return False
        
        print("✅ All LLM integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\n📝 Creating .env file from template...")
        
        try:
            env_content = env_example.read_text()
            env_file.write_text(env_content)
            print("✅ .env file created")
            print("📝 Edit .env file and set your OPENAI_API_KEY")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("⚠️  No .env.example template found")
        return False

def main():
    """Main setup function."""
    print("🚀 LLMFlow OpenAI Integration Setup")
    print("=" * 50)
    
    setup_success = True
    
    # Check requirements
    if not check_requirements():
        setup_success = False
    
    # Create .env file
    create_env_file()
    
    # Check API key
    if not check_api_key():
        setup_success = False
        print("\n⚠️  Continuing without API key (testing only)")
    
    # Test integration
    if not test_llm_integration():
        setup_success = False
    
    print("\n" + "=" * 50)
    
    if setup_success:
        print("🎉 SETUP COMPLETE!")
        print("✅ LLMFlow OpenAI integration is ready to use!")
        print("\n🚀 Next steps:")
        print("1. Start your LLMFlow system")
        print("2. Components will automatically use LLM optimization")
        print("3. Monitor optimization recommendations in logs")
        print("4. Check the LLM_INTEGRATION_GUIDE.md for advanced usage")
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("Please resolve the issues above and run setup again.")
        print("\n🆘 Common solutions:")
        print("- Install dependencies: pip install openai>=1.0.0 aiohttp")
        print("- Set API key: export OPENAI_API_KEY='your-key'")
        print("- Check network connection for API access")
    
    print("=" * 50)
    return setup_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
