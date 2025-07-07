"""
Message Signing Module

This module provides cryptographic signing functionality for LLMFlow messages.
"""

import logging
import hashlib
import hmac
from typing import Any, Dict, Optional
from datetime import datetime
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class MessageSigner:
    """Handles cryptographic signing of messages."""
    
    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
        self.hash_algorithm = getattr(hashes, algorithm.upper())()
    
    async def sign_message_hmac(self, message: bytes, secret_key: str) -> bytes:
        """Sign message using HMAC."""
        try:
            signature = hmac.new(
                secret_key.encode(),
                message,
                hashlib.sha256
            ).hexdigest()
            
            return signature.encode()
            
        except Exception as e:
            logger.error(f"HMAC signing failed: {e}")
            raise
    
    async def verify_message_hmac(self, message: bytes, signature: bytes, secret_key: str) -> bool:
        """Verify HMAC signature."""
        try:
            expected_signature = await self.sign_message_hmac(message, secret_key)
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"HMAC verification failed: {e}")
            return False
    
    async def sign_message_rsa(self, message: bytes, private_key: bytes) -> bytes:
        """Sign message using RSA private key."""
        try:
            # Load private key
            key = serialization.load_pem_private_key(
                private_key,
                password=None,
                backend=default_backend()
            )
            
            # Sign message
            signature = key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(self.hash_algorithm),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                self.hash_algorithm
            )
            
            return signature
            
        except Exception as e:
            logger.error(f"RSA signing failed: {e}")
            raise
    
    async def verify_message_rsa(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify RSA signature."""
        try:
            # Load public key
            key = serialization.load_pem_public_key(
                public_key,
                backend=default_backend()
            )
            
            # Verify signature
            key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(self.hash_algorithm),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                self.hash_algorithm
            )
            
            return True
            
        except Exception as e:
            logger.debug(f"RSA verification failed: {e}")
            return False
    
    async def generate_rsa_keypair(self, key_size: int = 2048) -> Dict[str, bytes]:
        """Generate RSA key pair."""
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            
            # Get public key
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return {
                "private_key": private_pem,
                "public_key": public_pem
            }
            
        except Exception as e:
            logger.error(f"RSA key generation failed: {e}")
            raise
    
    async def hash_message(self, message: bytes) -> str:
        """Calculate hash of message."""
        try:
            hash_obj = hashlib.sha256()
            hash_obj.update(message)
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Message hashing failed: {e}")
            raise


class SignatureManager:
    """Manages message signatures for LLMFlow."""
    
    def __init__(self):
        self.signer = MessageSigner()
        self.signing_keys: Dict[str, Dict[str, bytes]] = {}
        self.default_key_id: Optional[str] = None
    
    async def add_signing_key(self, key_id: str, private_key: bytes, public_key: bytes, 
                             is_default: bool = False) -> None:
        """Add a signing key pair."""
        self.signing_keys[key_id] = {
            "private_key": private_key,
            "public_key": public_key,
            "created_at": datetime.utcnow().isoformat().encode()
        }
        
        if is_default or self.default_key_id is None:
            self.default_key_id = key_id
        
        logger.info(f"Added signing key: {key_id}")
    
    async def generate_and_add_key(self, key_id: str, key_size: int = 2048, 
                                  is_default: bool = False) -> Dict[str, bytes]:
        """Generate and add a new RSA key pair."""
        keypair = await self.signer.generate_rsa_keypair(key_size)
        await self.add_signing_key(
            key_id, 
            keypair["private_key"], 
            keypair["public_key"],
            is_default
        )
        return keypair
    
    async def sign_message(self, message: bytes, key_id: str = None) -> Dict[str, Any]:
        """Sign a message and return signature metadata."""
        key_id = key_id or self.default_key_id
        
        if not key_id or key_id not in self.signing_keys:
            raise ValueError(f"Signing key '{key_id}' not found")
        
        private_key = self.signing_keys[key_id]["private_key"]
        
        # Sign message
        signature = await self.signer.sign_message_rsa(message, private_key)
        
        # Calculate message hash
        message_hash = await self.signer.hash_message(message)
        
        return {
            "signature": signature,
            "key_id": key_id,
            "algorithm": "rsa_pss_sha256",
            "message_hash": message_hash,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def verify_signature(self, message: bytes, signature_data: Dict[str, Any]) -> bool:
        """Verify a message signature."""
        try:
            key_id = signature_data.get("key_id")
            signature = signature_data.get("signature")
            
            if not key_id or not signature:
                logger.error("Missing key_id or signature in signature data")
                return False
            
            if key_id not in self.signing_keys:
                logger.error(f"Unknown signing key: {key_id}")
                return False
            
            public_key = self.signing_keys[key_id]["public_key"]
            
            # Verify signature
            is_valid = await self.signer.verify_message_rsa(message, signature, public_key)
            
            if is_valid:
                # Also verify message hash if present
                expected_hash = signature_data.get("message_hash")
                if expected_hash:
                    actual_hash = await self.signer.hash_message(message)
                    is_valid = actual_hash == expected_hash
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def get_public_key(self, key_id: str) -> Optional[bytes]:
        """Get public key for a key ID."""
        if key_id in self.signing_keys:
            return self.signing_keys[key_id]["public_key"]
        return None
    
    def list_keys(self) -> List[str]:
        """List available key IDs."""
        return list(self.signing_keys.keys())
    
    def get_default_key_id(self) -> Optional[str]:
        """Get default key ID."""
        return self.default_key_id


# Global signature manager
_signature_manager = SignatureManager()

def get_signature_manager() -> SignatureManager:
    """Get the global signature manager."""
    return _signature_manager
