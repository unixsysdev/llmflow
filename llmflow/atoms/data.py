"""
Data Atom Definitions

This module contains the built-in data atom types for the LLMFlow framework.
"""

import re
import json
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
import msgpack

from ..core.base import DataAtom, ValidationResult


class StringAtom(DataAtom):
    """Basic string data atom."""
    
    def __init__(self, value: str, metadata: Dict[str, Any] = None):
        if not isinstance(value, str):
            raise TypeError("StringAtom value must be a string")
        super().__init__(value, metadata)
    
    def validate(self) -> ValidationResult:
        """Validate the string value."""
        if not isinstance(self.value, str):
            return ValidationResult.error("Value must be a string")
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize the string atom."""
        data = {
            "type": "StringAtom",
            "value": self.value,
            "metadata": self.metadata,
            "atom_id": self.atom_id
        }
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'StringAtom':
        """Deserialize bytes to a string atom."""
        unpacked = msgpack.unpackb(data, raw=False)
        atom = cls(unpacked["value"], unpacked["metadata"])
        atom.atom_id = unpacked["atom_id"]
        return atom


class IntegerAtom(DataAtom):
    """Basic integer data atom."""
    
    def __init__(self, value: int, metadata: Dict[str, Any] = None):
        if not isinstance(value, int):
            raise TypeError("IntegerAtom value must be an integer")
        super().__init__(value, metadata)
    
    def validate(self) -> ValidationResult:
        """Validate the integer value."""
        if not isinstance(self.value, int):
            return ValidationResult.error("Value must be an integer")
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize the integer atom."""
        data = {
            "type": "IntegerAtom",
            "value": self.value,
            "metadata": self.metadata,
            "atom_id": self.atom_id
        }
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'IntegerAtom':
        """Deserialize bytes to an integer atom."""
        unpacked = msgpack.unpackb(data, raw=False)
        atom = cls(unpacked["value"], unpacked["metadata"])
        atom.atom_id = unpacked["atom_id"]
        return atom


class BooleanAtom(DataAtom):
    """Basic boolean data atom."""
    
    def __init__(self, value: bool, metadata: Dict[str, Any] = None):
        if not isinstance(value, bool):
            raise TypeError("BooleanAtom value must be a boolean")
        super().__init__(value, metadata)
    
    def validate(self) -> ValidationResult:
        """Validate the boolean value."""
        if not isinstance(self.value, bool):
            return ValidationResult.error("Value must be a boolean")
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize the boolean atom."""
        data = {
            "type": "BooleanAtom",
            "value": self.value,
            "metadata": self.metadata,
            "atom_id": self.atom_id
        }
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'BooleanAtom':
        """Deserialize bytes to a boolean atom."""
        unpacked = msgpack.unpackb(data, raw=False)
        atom = cls(unpacked["value"], unpacked["metadata"])
        atom.atom_id = unpacked["atom_id"]
        return atom


class EmailAtom(DataAtom):
    """Email address data atom with validation."""
    
    EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    def __init__(self, value: str, metadata: Dict[str, Any] = None):
        if not isinstance(value, str):
            raise TypeError("EmailAtom value must be a string")
        super().__init__(value, metadata)
    
    def validate(self) -> ValidationResult:
        """Validate the email address."""
        if not isinstance(self.value, str):
            return ValidationResult.error("Email must be a string")
        
        if not self.EMAIL_REGEX.match(self.value):
            return ValidationResult.error("Invalid email format")
        
        if len(self.value) > 254:
            return ValidationResult.error("Email address too long")
        
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize the email atom."""
        data = {
            "type": "EmailAtom",
            "value": self.value,
            "metadata": self.metadata,
            "atom_id": self.atom_id
        }
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'EmailAtom':
        """Deserialize bytes to an email atom."""
        unpacked = msgpack.unpackb(data, raw=False)
        atom = cls(unpacked["value"], unpacked["metadata"])
        atom.atom_id = unpacked["atom_id"]
        return atom


class PasswordAtom(DataAtom):
    """Password data atom with validation."""
    
    def __init__(self, value: str, metadata: Dict[str, Any] = None):
        if not isinstance(value, str):
            raise TypeError("PasswordAtom value must be a string")
        super().__init__(value, metadata)
    
    def validate(self) -> ValidationResult:
        """Validate the password."""
        if not isinstance(self.value, str):
            return ValidationResult.error("Password must be a string")
        
        if len(self.value) < 8:
            return ValidationResult.error("Password must be at least 8 characters")
        
        if not re.search(r'[A-Za-z]', self.value):
            return ValidationResult.error("Password must contain letters")
        
        if not re.search(r'[0-9]', self.value):
            return ValidationResult.error("Password must contain numbers")
        
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize the password atom."""
        data = {
            "type": "PasswordAtom",
            "value": self.value,
            "metadata": self.metadata,
            "atom_id": self.atom_id
        }
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'PasswordAtom':
        """Deserialize bytes to a password atom."""
        unpacked = msgpack.unpackb(data, raw=False)
        atom = cls(unpacked["value"], unpacked["metadata"])
        atom.atom_id = unpacked["atom_id"]
        return atom


class TimestampAtom(DataAtom):
    """Timestamp data atom."""
    
    def __init__(self, value: datetime = None, metadata: Dict[str, Any] = None):
        if value is None:
            value = datetime.utcnow()
        if not isinstance(value, datetime):
            raise TypeError("TimestampAtom value must be a datetime")
        super().__init__(value, metadata)
    
    def validate(self) -> ValidationResult:
        """Validate the timestamp."""
        if not isinstance(self.value, datetime):
            return ValidationResult.error("Value must be a datetime")
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize the timestamp atom."""
        data = {
            "type": "TimestampAtom",
            "value": self.value.isoformat(),
            "metadata": self.metadata,
            "atom_id": self.atom_id
        }
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'TimestampAtom':
        """Deserialize bytes to a timestamp atom."""
        unpacked = msgpack.unpackb(data, raw=False)
        timestamp = datetime.fromisoformat(unpacked["value"])
        atom = cls(timestamp, unpacked["metadata"])
        atom.atom_id = unpacked["atom_id"]
        return atom


class CurrencyAtom(DataAtom):
    """Currency amount data atom."""
    
    VALID_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY"}
    
    def __init__(self, amount: Union[Decimal, float, int], currency: str, metadata: Dict[str, Any] = None):
        if isinstance(amount, (float, int)):
            amount = Decimal(str(amount))
        elif not isinstance(amount, Decimal):
            raise TypeError("CurrencyAtom amount must be a Decimal, float, or int")
        
        if not isinstance(currency, str):
            raise TypeError("CurrencyAtom currency must be a string")
        
        value = {"amount": amount, "currency": currency.upper()}
        super().__init__(value, metadata)
    
    def validate(self) -> ValidationResult:
        """Validate the currency amount."""
        if not isinstance(self.value, dict):
            return ValidationResult.error("Value must be a dictionary")
        
        if "amount" not in self.value or "currency" not in self.value:
            return ValidationResult.error("Value must contain 'amount' and 'currency' keys")
        
        amount = self.value["amount"]
        currency = self.value["currency"]
        
        if not isinstance(amount, Decimal):
            return ValidationResult.error("Amount must be a Decimal")
        
        if amount < 0:
            return ValidationResult.error("Amount cannot be negative")
        
        if currency not in self.VALID_CURRENCIES:
            return ValidationResult.error(f"Currency must be one of {self.VALID_CURRENCIES}")
        
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize the currency atom."""
        data = {
            "type": "CurrencyAtom",
            "value": {
                "amount": str(self.value["amount"]),
                "currency": self.value["currency"]
            },
            "metadata": self.metadata,
            "atom_id": self.atom_id
        }
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'CurrencyAtom':
        """Deserialize bytes to a currency atom."""
        unpacked = msgpack.unpackb(data, raw=False)
        amount = Decimal(unpacked["value"]["amount"])
        currency = unpacked["value"]["currency"]
        atom = cls(amount, currency, unpacked["metadata"])
        atom.atom_id = unpacked["atom_id"]
        return atom


class ListAtom(DataAtom):
    """List data atom containing other atoms."""
    
    def __init__(self, items: List[DataAtom], metadata: Dict[str, Any] = None):
        if not isinstance(items, list):
            raise TypeError("ListAtom items must be a list")
        if not all(isinstance(item, DataAtom) for item in items):
            raise TypeError("All items in ListAtom must be DataAtom instances")
        super().__init__(items, metadata)
    
    def validate(self) -> ValidationResult:
        """Validate the list and all its items."""
        if not isinstance(self.value, list):
            return ValidationResult.error("Value must be a list")
        
        for i, item in enumerate(self.value):
            if not isinstance(item, DataAtom):
                return ValidationResult.error(f"Item {i} must be a DataAtom")
            
            item_result = item.validate()
            if not item_result.is_valid:
                return ValidationResult.error(f"Item {i} validation failed: {item_result.errors}")
        
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize the list atom."""
        data = {
            "type": "ListAtom",
            "value": [item.serialize() for item in self.value],
            "metadata": self.metadata,
            "atom_id": self.atom_id
        }
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'ListAtom':
        """Deserialize bytes to a list atom."""
        unpacked = msgpack.unpackb(data, raw=False)
        # Note: This is a simplified deserialization
        # In a full implementation, you'd need to properly deserialize each item
        # based on its type
        items = []  # Would deserialize each item properly
        atom = cls(items, unpacked["metadata"])
        atom.atom_id = unpacked["atom_id"]
        return atom


class UUIDAtom(DataAtom):
    """UUID data atom."""
    
    def __init__(self, value: Union[str, uuid.UUID] = None, metadata: Dict[str, Any] = None):
        if value is None:
            value = uuid.uuid4()
        elif isinstance(value, str):
            value = uuid.UUID(value)
        elif not isinstance(value, uuid.UUID):
            raise TypeError("UUIDAtom value must be a UUID or string")
        super().__init__(value, metadata)
    
    def validate(self) -> ValidationResult:
        """Validate the UUID."""
        if not isinstance(self.value, uuid.UUID):
            return ValidationResult.error("Value must be a UUID")
        return ValidationResult.success()
    
    def serialize(self) -> bytes:
        """Serialize the UUID atom."""
        data = {
            "type": "UUIDAtom",
            "value": str(self.value),
            "metadata": self.metadata,
            "atom_id": self.atom_id
        }
        return msgpack.packb(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'UUIDAtom':
        """Deserialize bytes to a UUID atom."""
        unpacked = msgpack.unpackb(data, raw=False)
        uuid_obj = uuid.UUID(unpacked["value"])
        atom = cls(uuid_obj, unpacked["metadata"])
        atom.atom_id = unpacked["atom_id"]
        return atom
