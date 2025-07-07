"""
Service Atom Definitions

This module contains built-in service atom types for the LLMFlow framework.
"""

import hashlib
import bcrypt
import jwt
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..core.base import ServiceAtom, DataAtom
from .data import (
    StringAtom, BooleanAtom, EmailAtom, PasswordAtom, 
    TimestampAtom, IntegerAtom, UUIDAtom
)


class ValidateEmailAtom(ServiceAtom):
    """Service atom for validating email addresses."""
    
    def __init__(self):
        super().__init__(
            name="validate-email",
            input_types=["llmflow.atoms.data.EmailAtom"],
            output_types=["llmflow.atoms.data.BooleanAtom"]
        )
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Validate an email address."""
        email_atom = inputs[0]
        validation_result = email_atom.validate()
        return [BooleanAtom(validation_result.is_valid)]


class HashPasswordAtom(ServiceAtom):
    """Service atom for hashing passwords."""
    
    def __init__(self):
        super().__init__(
            name="hash-password",
            input_types=["llmflow.atoms.data.PasswordAtom"],
            output_types=["llmflow.atoms.data.StringAtom"]
        )
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Hash a password using bcrypt."""
        password_atom = inputs[0]
        password_bytes = password_atom.value.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return [StringAtom(hashed.decode('utf-8'))]


class VerifyPasswordAtom(ServiceAtom):
    """Service atom for verifying passwords against hashes."""
    
    def __init__(self):
        super().__init__(
            name="verify-password",
            input_types=["llmflow.atoms.data.PasswordAtom", "llmflow.atoms.data.StringAtom"],
            output_types=["llmflow.atoms.data.BooleanAtom"]
        )
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Verify a password against a hash."""
        password_atom = inputs[0]
        hash_atom = inputs[1]
        
        password_bytes = password_atom.value.encode('utf-8')
        hash_bytes = hash_atom.value.encode('utf-8')
        
        is_valid = bcrypt.checkpw(password_bytes, hash_bytes)
        return [BooleanAtom(is_valid)]


class GenerateTokenAtom(ServiceAtom):
    """Service atom for generating JWT tokens."""
    
    def __init__(self, secret_key: str = "default-secret", expiration_hours: int = 24):
        super().__init__(
            name="generate-token",
            input_types=["llmflow.atoms.data.StringAtom"],  # user_id
            output_types=["llmflow.atoms.data.StringAtom"]   # jwt_token
        )
        self.secret_key = secret_key
        self.expiration_hours = expiration_hours
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Generate a JWT token for a user."""
        user_id_atom = inputs[0]
        user_id = user_id_atom.value
        
        now = datetime.utcnow()
        payload = {
            "user_id": user_id,
            "exp": now + timedelta(hours=self.expiration_hours),
            "iat": now,
            "iss": "llmflow"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return [StringAtom(token)]


class VerifyTokenAtom(ServiceAtom):
    """Service atom for verifying JWT tokens."""
    
    def __init__(self, secret_key: str = "default-secret"):
        super().__init__(
            name="verify-token",
            input_types=["llmflow.atoms.data.StringAtom"],  # jwt_token
            output_types=["llmflow.atoms.data.BooleanAtom", "llmflow.atoms.data.StringAtom"]  # is_valid, user_id
        )
        self.secret_key = secret_key
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Verify a JWT token."""
        token_atom = inputs[0]
        token = token_atom.value
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            user_id = payload.get("user_id", "")
            return [BooleanAtom(True), StringAtom(user_id)]
        except jwt.ExpiredSignatureError:
            return [BooleanAtom(False), StringAtom("Token expired")]
        except jwt.InvalidTokenError:
            return [BooleanAtom(False), StringAtom("Invalid token")]


class TimestampAtom(ServiceAtom):
    """Service atom for generating timestamps."""
    
    def __init__(self):
        super().__init__(
            name="generate-timestamp",
            input_types=[],
            output_types=["llmflow.atoms.data.TimestampAtom"]
        )
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Generate a current timestamp."""
        from .data import TimestampAtom
        return [TimestampAtom()]


class HashStringAtom(ServiceAtom):
    """Service atom for hashing strings with SHA-256."""
    
    def __init__(self):
        super().__init__(
            name="hash-string",
            input_types=["llmflow.atoms.data.StringAtom"],
            output_types=["llmflow.atoms.data.StringAtom"]
        )
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Hash a string using SHA-256."""
        string_atom = inputs[0]
        string_bytes = string_atom.value.encode('utf-8')
        hash_digest = hashlib.sha256(string_bytes).hexdigest()
        return [StringAtom(hash_digest)]


class GenerateUUIDAtom(ServiceAtom):
    """Service atom for generating UUIDs."""
    
    def __init__(self):
        super().__init__(
            name="generate-uuid",
            input_types=[],
            output_types=["llmflow.atoms.data.UUIDAtom"]
        )
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Generate a new UUID."""
        return [UUIDAtom()]


class ConcatenateStringsAtom(ServiceAtom):
    """Service atom for concatenating strings."""
    
    def __init__(self, separator: str = ""):
        super().__init__(
            name="concatenate-strings",
            input_types=["llmflow.atoms.data.StringAtom", "llmflow.atoms.data.StringAtom"],
            output_types=["llmflow.atoms.data.StringAtom"]
        )
        self.separator = separator
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Concatenate two strings."""
        string1 = inputs[0].value
        string2 = inputs[1].value
        result = string1 + self.separator + string2
        return [StringAtom(result)]


class CompareStringsAtom(ServiceAtom):
    """Service atom for comparing strings."""
    
    def __init__(self, case_sensitive: bool = True):
        super().__init__(
            name="compare-strings",
            input_types=["llmflow.atoms.data.StringAtom", "llmflow.atoms.data.StringAtom"],
            output_types=["llmflow.atoms.data.BooleanAtom"]
        )
        self.case_sensitive = case_sensitive
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Compare two strings for equality."""
        string1 = inputs[0].value
        string2 = inputs[1].value
        
        if not self.case_sensitive:
            string1 = string1.lower()
            string2 = string2.lower()
        
        are_equal = string1 == string2
        return [BooleanAtom(are_equal)]


class ValidateStringLengthAtom(ServiceAtom):
    """Service atom for validating string length."""
    
    def __init__(self, min_length: int = 0, max_length: int = None):
        super().__init__(
            name="validate-string-length",
            input_types=["llmflow.atoms.data.StringAtom"],
            output_types=["llmflow.atoms.data.BooleanAtom"]
        )
        self.min_length = min_length
        self.max_length = max_length
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Validate string length."""
        string_atom = inputs[0]
        string_value = string_atom.value
        
        if len(string_value) < self.min_length:
            return [BooleanAtom(False)]
        
        if self.max_length is not None and len(string_value) > self.max_length:
            return [BooleanAtom(False)]
        
        return [BooleanAtom(True)]


class AddIntegersAtom(ServiceAtom):
    """Service atom for adding integers."""
    
    def __init__(self):
        super().__init__(
            name="add-integers",
            input_types=["llmflow.atoms.data.IntegerAtom", "llmflow.atoms.data.IntegerAtom"],
            output_types=["llmflow.atoms.data.IntegerAtom"]
        )
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Add two integers."""
        int1 = inputs[0].value
        int2 = inputs[1].value
        result = int1 + int2
        return [IntegerAtom(result)]


class LogicAndAtom(ServiceAtom):
    """Service atom for logical AND operation."""
    
    def __init__(self):
        super().__init__(
            name="logic-and",
            input_types=["llmflow.atoms.data.BooleanAtom", "llmflow.atoms.data.BooleanAtom"],
            output_types=["llmflow.atoms.data.BooleanAtom"]
        )
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Perform logical AND on two boolean values."""
        bool1 = inputs[0].value
        bool2 = inputs[1].value
        result = bool1 and bool2
        return [BooleanAtom(result)]


class LogicOrAtom(ServiceAtom):
    """Service atom for logical OR operation."""
    
    def __init__(self):
        super().__init__(
            name="logic-or",
            input_types=["llmflow.atoms.data.BooleanAtom", "llmflow.atoms.data.BooleanAtom"],
            output_types=["llmflow.atoms.data.BooleanAtom"]
        )
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Perform logical OR on two boolean values."""
        bool1 = inputs[0].value
        bool2 = inputs[1].value
        result = bool1 or bool2
        return [BooleanAtom(result)]


class LogicNotAtom(ServiceAtom):
    """Service atom for logical NOT operation."""
    
    def __init__(self):
        super().__init__(
            name="logic-not",
            input_types=["llmflow.atoms.data.BooleanAtom"],
            output_types=["llmflow.atoms.data.BooleanAtom"]
        )
    
    def process(self, inputs: List[DataAtom]) -> List[DataAtom]:
        """Perform logical NOT on a boolean value."""
        bool_value = inputs[0].value
        result = not bool_value
        return [BooleanAtom(result)]
