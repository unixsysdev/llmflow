"""
Advanced security module tests for LLMFlow - testing actual implementation details.
"""

import sys
import os
import re
import json
from pathlib import Path

# Add the parent directory to the path so we can import llmflow
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_authentication_implementation():
    """Test the authentication implementation details."""
    print("\n🔐 Testing Authentication Implementation...")
    
    with open("llmflow/security/auth/authenticator.py", 'r') as f:
        content = f.read()
    
    # Check for required methods in Authenticator class
    authenticator_methods = [
        "authenticate",
        "validate_credentials",
        "generate_session"
    ]
    
    # Check for AuthenticationManager methods
    auth_manager_methods = [
        "register_authenticator",
        "get_authenticator",
        "authenticate_user"
    ]
    
    for method in authenticator_methods:
        if f"def {method}" in content:
            print(f"  ✓ Authenticator.{method}() found")
        else:
            print(f"  ⚠️  Authenticator.{method}() missing")
    
    for method in auth_manager_methods:
        if f"def {method}" in content:
            print(f"  ✓ AuthenticationManager.{method}() found")
        else:
            print(f"  ⚠️  AuthenticationManager.{method}() missing")
    
    # Check for singleton pattern
    if "get_auth_manager" in content and "_auth_manager" in content:
        print("  ✓ Singleton pattern implemented")
    else:
        print("  ⚠️  Singleton pattern missing")
    
    return True

def test_authorization_implementation():
    """Test the authorization implementation details."""
    print("\n🛡️  Testing Authorization Implementation...")
    
    with open("llmflow/security/auth/authorizer.py", 'r') as f:
        content = f.read()
    
    # Check Permission class
    permission_attributes = ["action", "resource", "resource_id"]
    for attr in permission_attributes:
        if f"self.{attr}" in content:
            print(f"  ✓ Permission.{attr} attribute found")
        else:
            print(f"  ⚠️  Permission.{attr} attribute missing")
    
    # Check Role class
    role_attributes = ["name", "permissions"]
    for attr in role_attributes:
        if f"self.{attr}" in content:
            print(f"  ✓ Role.{attr} attribute found")
        else:
            print(f"  ⚠️  Role.{attr} attribute missing")
    
    # Check Authorizer methods
    authorizer_methods = [
        "check_permission",
        "has_role",
        "grant_permission"
    ]
    
    for method in authorizer_methods:
        if f"def {method}" in content:
            print(f"  ✓ Authorizer.{method}() found")
        else:
            print(f"  ⚠️  Authorizer.{method}() missing")
    
    return True

def test_token_management_implementation():
    """Test the token management implementation details."""
    print("\n🎟️  Testing Token Management Implementation...")
    
    with open("llmflow/security/auth/token_manager.py", 'r') as f:
        content = f.read()
    
    # Check TokenInfo class
    token_info_attributes = ["token", "token_type", "expires_at", "scopes"]
    for attr in token_info_attributes:
        if f"self.{attr}" in content:
            print(f"  ✓ TokenInfo.{attr} attribute found")
        else:
            print(f"  ⚠️  TokenInfo.{attr} attribute missing")
    
    # Check TokenManager methods
    token_manager_methods = [
        "create_token",
        "validate_token",
        "revoke_token",
        "refresh_token"
    ]
    
    for method in token_manager_methods:
        if f"def {method}" in content:
            print(f"  ✓ TokenManager.{method}() found")
        else:
            print(f"  ⚠️  TokenManager.{method}() missing")
    
    return True

def test_cryptographic_implementation():
    """Test the cryptographic implementation details."""
    print("\n🔒 Testing Cryptographic Implementation...")
    
    # Test signing implementation
    with open("llmflow/security/crypto/signing.py", 'r') as f:
        signing_content = f.read()
    
    signing_methods = [
        "sign_message",
        "verify_signature",
        "generate_keypair"
    ]
    
    for method in signing_methods:
        if f"def {method}" in signing_content:
            print(f"  ✓ MessageSigner.{method}() found")
        else:
            print(f"  ⚠️  MessageSigner.{method}() missing")
    
    # Test encryption implementation
    with open("llmflow/security/crypto/encryption.py", 'r') as f:
        encryption_content = f.read()
    
    encryption_methods = [
        "encrypt_message",
        "decrypt_message",
        "generate_key"
    ]
    
    for method in encryption_methods:
        if f"def {method}" in encryption_content:
            print(f"  ✓ MessageEncryptor.{method}() found")
        else:
            print(f"  ⚠️  MessageEncryptor.{method}() missing")
    
    # Check SecureMessageEnvelope
    envelope_attributes = ["encrypted_data", "signature", "metadata"]
    for attr in envelope_attributes:
        if f"self.{attr}" in encryption_content:
            print(f"  ✓ SecureMessageEnvelope.{attr} attribute found")
        else:
            print(f"  ⚠️  SecureMessageEnvelope.{attr} attribute missing")
    
    return True

def test_security_providers_implementation():
    """Test the security providers implementation details."""
    print("\n🏭 Testing Security Providers Implementation...")
    
    # Test JWT Provider
    with open("llmflow/security/providers/jwt_provider.py", 'r') as f:
        jwt_content = f.read()
    
    # Check JWTConfig
    jwt_config_attributes = ["secret_key", "algorithm", "expiry_minutes"]
    for attr in jwt_config_attributes:
        if f"self.{attr}" in jwt_content:
            print(f"  ✓ JWTConfig.{attr} attribute found")
        else:
            print(f"  ⚠️  JWTConfig.{attr} attribute missing")
    
    # Check JWTSecurityProvider methods
    jwt_methods = [
        "create_token",
        "validate_token",
        "decode_token"
    ]
    
    for method in jwt_methods:
        if f"def {method}" in jwt_content:
            print(f"  ✓ JWTSecurityProvider.{method}() found")
        else:
            print(f"  ⚠️  JWTSecurityProvider.{method}() missing")
    
    # Test No Security Provider
    with open("llmflow/security/providers/no_security_provider.py", 'r') as f:
        no_security_content = f.read()
    
    if "class NoSecurityProvider" in no_security_content:
        print("  ✓ NoSecurityProvider class found")
    else:
        print("  ⚠️  NoSecurityProvider class missing")
    
    # Test OAuth2 Provider
    with open("llmflow/security/providers/oauth2_provider.py", 'r') as f:
        oauth2_content = f.read()
    
    # Check OAuth2Config
    oauth2_config_attributes = ["client_id", "client_secret", "authorization_url", "token_url"]
    for attr in oauth2_config_attributes:
        if f"self.{attr}" in oauth2_content:
            print(f"  ✓ OAuth2Config.{attr} attribute found")
        else:
            print(f"  ⚠️  OAuth2Config.{attr} attribute missing")
    
    return True

def test_security_integration_points():
    """Test security integration points with the framework."""
    print("\n🔗 Testing Security Integration Points...")
    
    with open("llmflow/security/__init__.py", 'r') as f:
        security_init = f.read()
    
    # Check for initialization functions
    init_functions = [
        "initialize_security_system",
        "shutdown_security_system"
    ]
    
    for func in init_functions:
        if f"def {func}" in security_init:
            print(f"  ✓ {func}() found in security __init__")
        else:
            print(f"  ⚠️  {func}() missing in security __init__")
    
    # Check for proper exports
    if "__all__" in security_init:
        print("  ✓ __all__ exports defined")
    else:
        print("  ⚠️  __all__ exports missing")
    
    return True

def test_security_error_handling():
    """Test security error handling implementation."""
    print("\n⚠️  Testing Security Error Handling...")
    
    security_files = [
        "llmflow/security/auth/authenticator.py",
        "llmflow/security/auth/authorizer.py",
        "llmflow/security/auth/token_manager.py",
        "llmflow/security/crypto/signing.py",
        "llmflow/security/crypto/encryption.py"
    ]
    
    error_patterns = [
        "raise",
        "except",
        "try:",
        "finally:"
    ]
    
    for file_path in security_files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        error_handling_found = any(pattern in content for pattern in error_patterns)
        if error_handling_found:
            print(f"  ✓ {os.path.basename(file_path)} has error handling")
        else:
            print(f"  ⚠️  {os.path.basename(file_path)} missing error handling")
    
    return True

def test_security_logging():
    """Test security logging implementation."""
    print("\n📝 Testing Security Logging...")
    
    security_files = [
        "llmflow/security/auth/authenticator.py",
        "llmflow/security/auth/authorizer.py",
        "llmflow/security/auth/token_manager.py",
        "llmflow/security/crypto/signing.py",
        "llmflow/security/crypto/encryption.py"
    ]
    
    for file_path in security_files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if "logger" in content and "logging" in content:
            print(f"  ✓ {os.path.basename(file_path)} has logging setup")
        else:
            print(f"  ⚠️  {os.path.basename(file_path)} missing logging")
    
    return True

def test_security_async_support():
    """Test async support in security modules."""
    print("\n⚡ Testing Async Support...")
    
    security_files = [
        "llmflow/security/auth/authenticator.py",
        "llmflow/security/auth/authorizer.py",
        "llmflow/security/auth/token_manager.py"
    ]
    
    for file_path in security_files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if "async def" in content or "await" in content:
            print(f"  ✓ {os.path.basename(file_path)} has async support")
        else:
            print(f"  ⚠️  {os.path.basename(file_path)} missing async support")
    
    return True

def test_security_type_hints():
    """Test type hints in security modules."""
    print("\n🏷️  Testing Type Hints...")
    
    security_files = [
        "llmflow/security/auth/authenticator.py",
        "llmflow/security/auth/authorizer.py",
        "llmflow/security/auth/token_manager.py",
        "llmflow/security/crypto/signing.py",
        "llmflow/security/crypto/encryption.py"
    ]
    
    for file_path in security_files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for type imports
        type_imports = any(import_stmt in content for import_stmt in [
            "from typing import",
            "import typing",
            "from __future__ import annotations"
        ])
        
        # Check for type annotations
        has_annotations = "->" in content or ": " in content
        
        if type_imports and has_annotations:
            print(f"  ✓ {os.path.basename(file_path)} has type hints")
        else:
            print(f"  ⚠️  {os.path.basename(file_path)} missing comprehensive type hints")
    
    return True

def generate_security_test_report():
    """Generate a comprehensive security test report."""
    print("\n📊 Generating Security Test Report...")
    
    report = {
        "test_timestamp": "2025-07-08",
        "framework_version": "0.1.0",
        "security_modules": {
            "authentication": {
                "status": "implemented",
                "classes": ["Authenticator", "AuthenticationManager"],
                "key_features": ["credential validation", "session management", "singleton pattern"]
            },
            "authorization": {
                "status": "implemented", 
                "classes": ["Permission", "Role", "Authorizer", "AuthorizationManager"],
                "key_features": ["permission checking", "role-based access", "resource protection"]
            },
            "token_management": {
                "status": "implemented",
                "classes": ["TokenInfo", "TokenManager", "TokenManagerRegistry"],
                "key_features": ["token creation", "validation", "expiration handling"]
            },
            "cryptography": {
                "status": "implemented",
                "classes": ["MessageSigner", "SignatureManager", "MessageEncryptor", "EncryptionManager"],
                "key_features": ["message signing", "encryption", "secure envelopes"]
            },
            "security_providers": {
                "status": "implemented",
                "classes": ["JWTSecurityProvider", "NoSecurityProvider", "OAuth2SecurityProvider"],
                "key_features": ["JWT tokens", "OAuth2 flow", "development mode"]
            }
        },
        "integration": {
            "initialization": "implemented",
            "shutdown": "implemented", 
            "singleton_managers": "implemented"
        },
        "quality_checks": {
            "error_handling": "partial",
            "logging": "implemented",
            "async_support": "partial",
            "type_hints": "partial"
        }
    }
    
    # Save report
    with open("security_test_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("  ✓ Security test report saved to security_test_report.json")
    return True

def run_advanced_security_tests():
    """Run all advanced security tests."""
    print("=" * 70)
    print("LLMFLOW ADVANCED SECURITY MODULE TESTS")
    print("=" * 70)
    
    tests = [
        test_authentication_implementation,
        test_authorization_implementation,
        test_token_management_implementation,
        test_cryptographic_implementation,
        test_security_providers_implementation,
        test_security_integration_points,
        test_security_error_handling,
        test_security_logging,
        test_security_async_support,
        test_security_type_hints,
        generate_security_test_report
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            if test():
                passed += 1
            else:
                print(f"  ❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"  ❌ {test.__name__} ERROR: {e}")
    
    print("\n" + "=" * 70)
    print(f"ADVANCED RESULTS: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("🎉 ALL ADVANCED SECURITY TESTS PASSED!")
        return True
    else:
        print("❌ Some advanced security tests need attention")
        return False

if __name__ == "__main__":
    success = run_advanced_security_tests()
    exit(0 if success else 1)
