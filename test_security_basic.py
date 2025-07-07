"""
Basic structure test for LLMFlow security module without external dependencies.
"""

import sys
import os

# Add the parent directory to the path so we can import llmflow
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_security_module_structure():
    """Test that security module files exist and have basic structure."""
    
    # Test file existence
    security_files = [
        "llmflow/security/__init__.py",
        "llmflow/security/auth/__init__.py",
        "llmflow/security/auth/authenticator.py",
        "llmflow/security/auth/authorizer.py",
        "llmflow/security/auth/token_manager.py",
        "llmflow/security/crypto/__init__.py",
        "llmflow/security/crypto/signing.py",
        "llmflow/security/crypto/encryption.py",
        "llmflow/security/providers/__init__.py",
        "llmflow/security/providers/jwt_provider.py",
        "llmflow/security/providers/no_security_provider.py",
        "llmflow/security/providers/oauth2_provider.py",
    ]
    
    missing_files = []
    for file_path in security_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"FAIL: Missing security files: {missing_files}")
        return False
    else:
        print("PASS: All security module files exist")
    
    return True

def test_security_module_imports():
    """Test basic imports without full dependency resolution."""
    
    try:
        # Test basic file reading to check syntax
        security_init_path = "llmflow/security/__init__.py"
        with open(security_init_path, 'r') as f:
            content = f.read()
            if "initialize_security_system" in content and "shutdown_security_system" in content:
                print("PASS: Security __init__.py has expected functions")
            else:
                print("FAIL: Security __init__.py missing expected functions")
                return False
                
        # Test auth files
        auth_files = [
            "llmflow/security/auth/authenticator.py",
            "llmflow/security/auth/authorizer.py", 
            "llmflow/security/auth/token_manager.py"
        ]
        
        for auth_file in auth_files:
            with open(auth_file, 'r') as f:
                content = f.read()
                if "class" in content and "def" in content:
                    print(f"PASS: {auth_file} has class and method definitions")
                else:
                    print(f"FAIL: {auth_file} missing class or method definitions")
                    return False
        
        # Test crypto files
        crypto_files = [
            "llmflow/security/crypto/signing.py",
            "llmflow/security/crypto/encryption.py"
        ]
        
        for crypto_file in crypto_files:
            with open(crypto_file, 'r') as f:
                content = f.read()
                if "class" in content and "def" in content:
                    print(f"PASS: {crypto_file} has class and method definitions")
                else:
                    print(f"FAIL: {crypto_file} missing class or method definitions")
                    return False
        
        # Test provider files
        provider_files = [
            "llmflow/security/providers/jwt_provider.py",
            "llmflow/security/providers/no_security_provider.py",
            "llmflow/security/providers/oauth2_provider.py"
        ]
        
        for provider_file in provider_files:
            with open(provider_file, 'r') as f:
                content = f.read()
                if "class" in content and "Provider" in content:
                    print(f"PASS: {provider_file} has Provider class")
                else:
                    print(f"FAIL: {provider_file} missing Provider class")
                    return False
                    
        print("PASS: All security modules have expected structure")
        return True
        
    except Exception as e:
        print(f"FAIL: Error testing security module imports: {e}")
        return False

def test_security_module_interfaces():
    """Test that security modules have consistent interfaces."""
    
    try:
        # Check authenticator interface
        with open("llmflow/security/auth/authenticator.py", 'r') as f:
            content = f.read()
            if "class Authenticator" in content and "class AuthenticationManager" in content:
                print("PASS: Authenticator has expected classes")
            else:
                print("FAIL: Authenticator missing expected classes")
                return False
        
        # Check authorizer interface
        with open("llmflow/security/auth/authorizer.py", 'r') as f:
            content = f.read()
            if "class Permission" in content and "class Role" in content and "class Authorizer" in content:
                print("PASS: Authorizer has expected classes")
            else:
                print("FAIL: Authorizer missing expected classes")
                return False
        
        # Check token manager interface
        with open("llmflow/security/auth/token_manager.py", 'r') as f:
            content = f.read()
            if "class TokenInfo" in content and "class TokenManager" in content:
                print("PASS: TokenManager has expected classes")
            else:
                print("FAIL: TokenManager missing expected classes")
                return False
        
        # Check signing interface
        with open("llmflow/security/crypto/signing.py", 'r') as f:
            content = f.read()
            if "class MessageSigner" in content and "class SignatureManager" in content:
                print("PASS: Signing module has expected classes")
            else:
                print("FAIL: Signing module missing expected classes")
                return False
        
        # Check encryption interface
        with open("llmflow/security/crypto/encryption.py", 'r') as f:
            content = f.read()
            if "class MessageEncryptor" in content and "class EncryptionManager" in content:
                print("PASS: Encryption module has expected classes")
            else:
                print("FAIL: Encryption module missing expected classes")
                return False
                
        print("PASS: All security modules have consistent interfaces")
        return True
        
    except Exception as e:
        print(f"FAIL: Error testing security module interfaces: {e}")
        return False

def run_security_tests():
    """Run all security tests."""
    print("=" * 60)
    print("LLMFLOW SECURITY MODULE TESTS")
    print("=" * 60)
    
    tests = [
        test_security_module_structure,
        test_security_module_imports,
        test_security_module_interfaces
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"  {test.__name__} FAILED")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL SECURITY TESTS PASSED!")
        return True
    else:
        print("❌ Some security tests failed")
        return False

if __name__ == "__main__":
    success = run_security_tests()
    exit(0 if success else 1)
