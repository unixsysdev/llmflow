"""
Message Serialization Interface

This module defines the interface for message serializers in the LLMFlow framework.
Serializers handle the conversion of data structures to and from binary formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Supported serialization formats."""
    MESSAGEPACK = "messagepack"
    JSON = "json"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    BSON = "bson"
    PICKLE = "pickle"
    CUSTOM = "custom"


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"
    SNAPPY = "snappy"


class SerializationError(Exception):
    """Base exception for serialization-related errors."""
    pass


class SerializationFormatError(SerializationError):
    """Raised when serialization format is not supported."""
    pass


class DeserializationError(SerializationError):
    """Raised when deserialization fails."""
    pass


class SchemaError(SerializationError):
    """Raised when schema validation fails."""
    pass


class SerializationMetadata:
    """
    Metadata about serialized data.
    """
    
    def __init__(self, 
                 format: SerializationFormat,
                 schema_version: Optional[str] = None,
                 compression: CompressionType = CompressionType.NONE,
                 content_type: Optional[str] = None,
                 size_bytes: Optional[int] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.format = format
        self.schema_version = schema_version
        self.compression = compression
        self.content_type = content_type
        self.size_bytes = size_bytes
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'format': self.format.value,
            'schema_version': self.schema_version,
            'compression': self.compression.value,
            'content_type': self.content_type,
            'size_bytes': self.size_bytes,
            'metadata': self.metadata
        }


class IMessageSerializer(ABC):
    """
    Interface for message serializers in LLMFlow.
    
    This interface defines the contract for all serialization implementations,
    including MessagePack, JSON, Protocol Buffers, and other formats.
    """
    
    @abstractmethod
    def get_format(self) -> SerializationFormat:
        """
        Get the serialization format.
        
        Returns:
            The serialization format this serializer supports
        """
        pass
    
    @abstractmethod
    def get_content_type(self) -> str:
        """
        Get the MIME content type for this serialization format.
        
        Returns:
            MIME content type string
        """
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """
        Get the file extension for this serialization format.
        
        Returns:
            File extension (without dot)
        """
        pass
    
    @abstractmethod
    def supports_schema_evolution(self) -> bool:
        """
        Check if this serializer supports schema evolution.
        
        Returns:
            True if schema evolution is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def supports_compression(self) -> bool:
        """
        Check if this serializer supports compression.
        
        Returns:
            True if compression is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_compression_types(self) -> List[CompressionType]:
        """
        Get the list of supported compression types.
        
        Returns:
            List of supported compression types
        """
        pass
    
    @abstractmethod
    async def serialize(self, 
                       data: Any, 
                       schema_version: Optional[str] = None,
                       compression: CompressionType = CompressionType.NONE) -> bytes:
        """
        Serialize data to bytes.
        
        Args:
            data: Data to serialize
            schema_version: Optional schema version
            compression: Compression type to use
            
        Returns:
            Serialized data as bytes
            
        Raises:
            SerializationError: If serialization fails
        """
        pass
    
    @abstractmethod
    async def deserialize(self, 
                         data: bytes, 
                         expected_type: Optional[Type] = None,
                         schema_version: Optional[str] = None) -> Any:
        """
        Deserialize data from bytes.
        
        Args:
            data: Serialized data
            expected_type: Expected type of deserialized data
            schema_version: Optional schema version
            
        Returns:
            Deserialized data
            
        Raises:
            DeserializationError: If deserialization fails
        """
        pass
    
    @abstractmethod
    async def get_metadata(self, data: bytes) -> SerializationMetadata:
        """
        Get metadata about serialized data.
        
        Args:
            data: Serialized data
            
        Returns:
            Metadata about the serialized data
            
        Raises:
            SerializationError: If metadata extraction fails
        """
        pass
    
    @abstractmethod
    async def validate_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            
        Returns:
            True if data is valid, False otherwise
            
        Raises:
            SchemaError: If schema validation fails
        """
        pass
    
    @abstractmethod
    async def register_schema(self, schema_id: str, schema: Dict[str, Any]) -> None:
        """
        Register a schema for use with this serializer.
        
        Args:
            schema_id: Unique identifier for the schema
            schema: Schema definition
            
        Raises:
            SchemaError: If schema registration fails
        """
        pass
    
    @abstractmethod
    async def get_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered schema.
        
        Args:
            schema_id: Schema identifier
            
        Returns:
            Schema definition or None if not found
            
        Raises:
            SchemaError: If schema retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_schema_version(self) -> str:
        """
        Get the current schema version.
        
        Returns:
            Current schema version
        """
        pass
    
    @abstractmethod
    async def set_schema_version(self, version: str) -> None:
        """
        Set the schema version.
        
        Args:
            version: Schema version to set
            
        Raises:
            SchemaError: If schema version setting fails
        """
        pass
    
    @abstractmethod
    async def compress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """
        Compress data using the specified compression type.
        
        Args:
            data: Data to compress
            compression_type: Compression type to use
            
        Returns:
            Compressed data
            
        Raises:
            SerializationError: If compression fails
        """
        pass
    
    @abstractmethod
    async def decompress(self, compressed_data: bytes, compression_type: CompressionType) -> bytes:
        """
        Decompress data using the specified compression type.
        
        Args:
            compressed_data: Compressed data
            compression_type: Compression type used
            
        Returns:
            Decompressed data
            
        Raises:
            SerializationError: If decompression fails
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get serialization statistics.
        
        Returns:
            Dictionary containing serialization statistics
        """
        pass


class SerializerRegistry:
    """
    Registry for managing multiple serializers.
    """
    
    def __init__(self):
        self.serializers: Dict[SerializationFormat, IMessageSerializer] = {}
        self.default_serializer: Optional[IMessageSerializer] = None
    
    def register(self, serializer: IMessageSerializer, is_default: bool = False) -> None:
        """
        Register a serializer.
        
        Args:
            serializer: Serializer to register
            is_default: Whether this should be the default serializer
        """
        format = serializer.get_format()
        self.serializers[format] = serializer
        
        if is_default or self.default_serializer is None:
            self.default_serializer = serializer
    
    def get_serializer(self, format: SerializationFormat) -> Optional[IMessageSerializer]:
        """
        Get a serializer by format.
        
        Args:
            format: Serialization format
            
        Returns:
            Serializer instance or None if not found
        """
        return self.serializers.get(format)
    
    def get_default_serializer(self) -> Optional[IMessageSerializer]:
        """
        Get the default serializer.
        
        Returns:
            Default serializer or None if not set
        """
        return self.default_serializer
    
    def get_supported_formats(self) -> List[SerializationFormat]:
        """
        Get the list of supported formats.
        
        Returns:
            List of supported serialization formats
        """
        return list(self.serializers.keys())
    
    def unregister(self, format: SerializationFormat) -> bool:
        """
        Unregister a serializer.
        
        Args:
            format: Serialization format to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if format in self.serializers:
            serializer = self.serializers[format]
            del self.serializers[format]
            
            # If this was the default serializer, pick a new default
            if self.default_serializer == serializer:
                self.default_serializer = next(iter(self.serializers.values()), None)
            
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary containing registry statistics
        """
        return {
            'registered_formats': [f.value for f in self.serializers.keys()],
            'default_format': self.default_serializer.get_format().value if self.default_serializer else None,
            'total_serializers': len(self.serializers)
        }
