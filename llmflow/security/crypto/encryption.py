"""
Message Encryption Module

This module provides encryption functionality for LLMFlow messages.
"""

import logging
import os
from typing import Any, Dict, Optional, List
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

logger = logging.getLogger(__name__)


class MessageEncryptor:
    """Handles encryption and decryption of messages."""
    
    def __init__(self):
        pass
    
    async def encrypt_symmetric(self, message: bytes, key: bytes) -> bytes:
        """Encrypt message using symmetric encryption (Fernet)."""
        try:
            fernet = Fernet(key)
            encrypted_message = fernet.encrypt(message)
            return encrypted_message
            
        except Exception as e:
            logger.error(f"Symmetric encryption failed: {e}")
            raise
    
    async def decrypt_symmetric(self, encrypted_message: bytes, key: bytes) -> bytes:
        """Decrypt message using symmetric encryption (Fernet)."""
        try:
            fernet = Fernet(key)
            decrypted_message = fernet.decrypt(encrypted_message)
            return decrypted_message
            
        except Exception as e:
            logger.error(f"Symmetric decryption failed: {e}")
            raise
    
    async def encrypt_asymmetric(self, message: bytes, public_key: bytes) -> bytes:
        """Encrypt message using RSA public key."""
        try:
            # Load public key
            key = serialization.load_pem_public_key(
                public_key,
                backend=default_backend()
            )
            
            # Encrypt message
            encrypted_message = key.encrypt(
                message,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return encrypted_message
            
        except Exception as e:
            logger.error(f"Asymmetric encryption failed: {e}")
            raise
    
    async def decrypt_asymmetric(self, encrypted_message: bytes, private_key: bytes) -> bytes:
        """Decrypt message using RSA private key."""
        try:
            # Load private key
            key = serialization.load_pem_private_key(
                private_key,
                password=None,
                backend=default_backend()
            )
            
            # Decrypt message
            decrypted_message = key.decrypt(
                encrypted_message,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted_message
            
        except Exception as e:
            logger.error(f"Asymmetric decryption failed: {e}")
            raise
    
    async def generate_symmetric_key(self) -> bytes:
        """Generate a symmetric encryption key."""
        return Fernet.generate_key()
    
    async def derive_key_from_password(self, password: str, salt: bytes = None) -> bytes:
        """Derive encryption key from password."""
        try:
            if salt is None:
                salt = os.urandom(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            return key
            
        except Exception as e:
            logger.error(f"Key derivation failed: {e}")
            raise


class EncryptionManager:
    """Manages encryption for LLMFlow messages."""
    
    def __init__(self):
        self.encryptor = MessageEncryptor()
        self.encryption_keys: Dict[str, Dict[str, Any]] = {}
        self.default_key_id: Optional[str] = None
    
    async def add_symmetric_key(self, key_id: str, key: bytes = None, is_default: bool = False) -> bytes:
        """Add a symmetric encryption key."""
        if key is None:
            key = await self.encryptor.generate_symmetric_key()
        
        self.encryption_keys[key_id] = {
            "type": "symmetric",
            "key": key,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if is_default or self.default_key_id is None:
            self.default_key_id = key_id
        
        logger.info(f"Added symmetric encryption key: {key_id}")
        return key
    
    async def add_asymmetric_keys(self, key_id: str, private_key: bytes, public_key: bytes,
                                 is_default: bool = False) -> None:
        """Add asymmetric encryption key pair."""
        self.encryption_keys[key_id] = {
            "type": "asymmetric",
            "private_key": private_key,
            "public_key": public_key,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if is_default or self.default_key_id is None:
            self.default_key_id = key_id
        
        logger.info(f"Added asymmetric encryption keys: {key_id}")
    
    async def encrypt_message(self, message: bytes, key_id: str = None, 
                             encryption_type: str = "symmetric") -> Dict[str, Any]:
        """Encrypt a message."""
        key_id = key_id or self.default_key_id
        
        if not key_id or key_id not in self.encryption_keys:
            raise ValueError(f"Encryption key '{key_id}' not found")
        
        key_data = self.encryption_keys[key_id]
        
        if encryption_type == "symmetric" or key_data["type"] == "symmetric":
            if key_data["type"] != "symmetric":
                raise ValueError(f"Key '{key_id}' is not a symmetric key")
            
            encrypted_data = await self.encryptor.encrypt_symmetric(message, key_data["key"])
            
            return {
                "encrypted_data": encrypted_data,
                "key_id": key_id,
                "encryption_type": "symmetric",
                "algorithm": "fernet",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif encryption_type == "asymmetric" or key_data["type"] == "asymmetric":
            if key_data["type"] != "asymmetric":
                raise ValueError(f"Key '{key_id}' is not an asymmetric key")
            
            encrypted_data = await self.encryptor.encrypt_asymmetric(message, key_data["public_key"])
            
            return {
                "encrypted_data": encrypted_data,
                "key_id": key_id,
                "encryption_type": "asymmetric",
                "algorithm": "rsa_oaep",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        else:
            raise ValueError(f"Unsupported encryption type: {encryption_type}")
    
    async def decrypt_message(self, encryption_data: Dict[str, Any]) -> bytes:
        """Decrypt a message."""
        try:
            key_id = encryption_data.get("key_id")
            encrypted_data = encryption_data.get("encrypted_data")
            encryption_type = encryption_data.get("encryption_type")
            
            if not key_id or not encrypted_data or not encryption_type:
                raise ValueError("Missing required encryption data")
            
            if key_id not in self.encryption_keys:
                raise ValueError(f"Encryption key '{key_id}' not found")
            
            key_data = self.encryption_keys[key_id]
            
            if encryption_type == "symmetric":
                decrypted_message = await self.encryptor.decrypt_symmetric(
                    encrypted_data, key_data["key"]
                )
            elif encryption_type == "asymmetric":
                decrypted_message = await self.encryptor.decrypt_asymmetric(
                    encrypted_data, key_data["private_key"]
                )
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
            
            return decrypted_message
            
        except Exception as e:
            logger.error(f"Message decryption failed: {e}")
            raise
    
    async def generate_and_add_key_pair(self, key_id: str, key_size: int = 2048,
                                       is_default: bool = False) -> Dict[str, bytes]:
        """Generate and add new RSA key pair."""
        # Reuse RSA generation from signing module
        from .signing import MessageSigner
        signer = MessageSigner()
        keypair = await signer.generate_rsa_keypair(key_size)
        
        await self.add_asymmetric_keys(
            key_id,
            keypair["private_key"],
            keypair["public_key"],
            is_default
        )
        
        return keypair
    
    def get_public_key(self, key_id: str) -> Optional[bytes]:
        """Get public key for asymmetric encryption."""
        if key_id in self.encryption_keys:
            key_data = self.encryption_keys[key_id]
            if key_data["type"] == "asymmetric":
                return key_data["public_key"]
        return None
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List available encryption keys."""
        return [
            {
                "key_id": key_id,
                "type": data["type"],
                "created_at": data["created_at"]
            }
            for key_id, data in self.encryption_keys.items()
        ]
    
    def get_default_key_id(self) -> Optional[str]:
        """Get default encryption key ID."""
        return self.default_key_id
    
    async def rotate_key(self, key_id: str, new_key: bytes = None) -> bytes:
        """Rotate an encryption key."""
        if key_id not in self.encryption_keys:
            raise ValueError(f"Key '{key_id}' not found")
        
        key_data = self.encryption_keys[key_id]
        
        if key_data["type"] == "symmetric":
            if new_key is None:
                new_key = await self.encryptor.generate_symmetric_key()
            
            # Store old key for migration period
            old_key = key_data["key"]
            key_data["old_key"] = old_key
            key_data["key"] = new_key
            key_data["rotated_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Rotated symmetric key: {key_id}")
            return new_key
        
        else:
            # For asymmetric keys, generate new pair
            keypair = await self.generate_and_add_key_pair(f"{key_id}_new")
            
            # Mark old key as deprecated
            key_data["deprecated_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Rotated asymmetric key: {key_id}")
            return keypair["public_key"]


class SecureMessageEnvelope:
    """Secure message envelope with encryption and signing."""
    
    def __init__(self, encryption_manager: EncryptionManager, signature_manager):
        self.encryption_manager = encryption_manager
        self.signature_manager = signature_manager
    
    async def seal_message(self, message: bytes, encrypt_key_id: str = None, 
                          sign_key_id: str = None) -> Dict[str, Any]:
        """Encrypt and sign a message."""
        # First encrypt the message
        encryption_data = await self.encryption_manager.encrypt_message(message, encrypt_key_id)
        
        # Then sign the encrypted data
        signature_data = await self.signature_manager.sign_message(
            encryption_data["encrypted_data"], sign_key_id
        )
        
        return {
            "envelope_version": "1.0",
            "encryption": encryption_data,
            "signature": signature_data,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def open_message(self, envelope: Dict[str, Any]) -> bytes:
        """Verify signature and decrypt a message."""
        # First verify the signature
        encrypted_data = envelope["encryption"]["encrypted_data"]
        signature_data = envelope["signature"]
        
        is_valid = await self.signature_manager.verify_signature(encrypted_data, signature_data)
        if not is_valid:
            raise ValueError("Invalid message signature")
        
        # Then decrypt the message
        decrypted_message = await self.encryption_manager.decrypt_message(envelope["encryption"])
        
        return decrypted_message


# Global encryption manager
_encryption_manager = EncryptionManager()

def get_encryption_manager() -> EncryptionManager:
    """Get the global encryption manager."""
    return _encryption_manager
