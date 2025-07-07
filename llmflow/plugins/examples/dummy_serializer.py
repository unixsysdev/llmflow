"""
Example Serializer Plugin - Dummy JSON Serializer

This is a simple example plugin that demonstrates how to implement
the IMessageSerializer interface for LLMFlow.
"""

import json
import gzip
import logging
from typing import Any, Dict, List, Optional, Type

from ..interfaces.base import Plugin, PluginStatus
from ..interfaces.serialization import (
    IMessageSerializer, SerializationFormat, CompressionType,
    SerializationError, DeserializationError, SchemaError,
    SerializationMetadata
)

logger = logging.getLogger(__name__)


class DummyJSONSerializer(Plugin, IMessageSerializer):
    """
    Dummy JSON serializer plugin for testing and demonstration.
    
    This plugin implements a basic JSON serializer that can be used
    for testing the plugin system with human-readable serialization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        Plugin.__init__(self, config)
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.current_schema_version = "1.0.0"
        self.stats = {
            'serializations': 0,
            'deserializations': 0,
            'compressions': 0,
            'decompressions': 0,
            'errors': 0
        }
    
    # Plugin interface methods
    def get_name(self) -> str:
        return "dummy_json_serializer"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "Dummy JSON serializer plugin for testing and demonstration"
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def get_interfaces(self) -> List[Type]:
        return [IMessageSerializer]
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the serializer plugin."""
        self.status = PluginStatus.INITIALIZING
        try:
            self.config.update(config)
            self.current_schema_version = self.config.get('schema_version', '1.0.0')
            self.status = PluginStatus.INITIALIZED
            logger.info("Dummy JSON serializer initialized")
        except Exception as e:
            self.status = PluginStatus.ERROR
            raise e
    
    async def start(self) -> None:
        """Start the serializer plugin."""
        self.status = PluginStatus.STARTING
        try:
            self.status = PluginStatus.RUNNING
            logger.info("Dummy JSON serializer started")
        except Exception as e:
            self.status = PluginStatus.ERROR
            raise e
    
    async def stop(self) -> None:
        """Stop the serializer plugin."""
        self.status = PluginStatus.STOPPING
        try:
            self.status = PluginStatus.STOPPED
            logger.info("Dummy JSON serializer stopped")
        except Exception as e:
            self.status = PluginStatus.ERROR
            raise e
    
    async def shutdown(self) -> None:
        """Shutdown the serializer plugin."""
        await self.stop()
        logger.info("Dummy JSON serializer shutdown")
    
    async def health_check(self) -> bool:
        """Check if the serializer is healthy."""
        return self.status == PluginStatus.RUNNING
    
    # Serializer interface methods
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.JSON
    
    def get_content_type(self) -> str:
        return "application/json"
    
    def get_file_extension(self) -> str:
        return "json"
    
    def supports_schema_evolution(self) -> bool:
        return True
    
    def supports_compression(self) -> bool:
        return True
    
    def get_supported_compression_types(self) -> List[CompressionType]:
        return [CompressionType.NONE, CompressionType.GZIP]
    
    async def serialize(self, 
                       data: Any, 
                       schema_version: Optional[str] = None,
                       compression: CompressionType = CompressionType.NONE) -> bytes:
        """Serialize data to bytes."""
        try:
            # Convert data to JSON
            json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            json_bytes = json_str.encode('utf-8')
            
            # Apply compression if requested
            if compression == CompressionType.GZIP:
                json_bytes = await self.compress(json_bytes, compression)
                self.stats['compressions'] += 1
            
            # Add metadata header
            metadata = {
                'format': self.get_format().value,
                'schema_version': schema_version or self.current_schema_version,
                'compression': compression.value,
                'content_type': self.get_content_type(),
                'size_bytes': len(json_bytes)
            }
            
            # Create final payload with metadata
            payload = {
                'metadata': metadata,
                'data': json_bytes.decode('utf-8') if compression == CompressionType.NONE else json_bytes.hex()
            }
            
            result = json.dumps(payload).encode('utf-8')
            
            self.stats['serializations'] += 1
            logger.debug(f"Serialized {len(result)} bytes")
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Serialization failed: {e}")
            raise SerializationError(str(e))
    
    async def deserialize(self, 
                         data: bytes, 
                         expected_type: Optional[Type] = None,
                         schema_version: Optional[str] = None) -> Any:
        """Deserialize data from bytes."""
        try:
            # Parse outer JSON
            payload = json.loads(data.decode('utf-8'))
            
            metadata = payload.get('metadata', {})
            data_str = payload.get('data', '')
            
            # Check format
            if metadata.get('format') != self.get_format().value:
                raise DeserializationError(f"Invalid format: {metadata.get('format')}")
            
            # Get data bytes
            compression = CompressionType(metadata.get('compression', 'none'))
            if compression == CompressionType.NONE:
                data_bytes = data_str.encode('utf-8')
            else:
                data_bytes = bytes.fromhex(data_str)
            
            # Decompress if needed
            if compression == CompressionType.GZIP:
                data_bytes = await self.decompress(data_bytes, compression)
                self.stats['decompressions'] += 1
            
            # Parse JSON data
            json_str = data_bytes.decode('utf-8')
            result = json.loads(json_str)
            
            # Type checking if requested
            if expected_type and not isinstance(result, expected_type):
                logger.warning(f"Expected {expected_type}, got {type(result)}")
            
            self.stats['deserializations'] += 1
            logger.debug(f"Deserialized {len(data)} bytes")
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Deserialization failed: {e}")
            raise DeserializationError(str(e))
    
    async def get_metadata(self, data: bytes) -> SerializationMetadata:
        """Get metadata about serialized data."""
        try:
            payload = json.loads(data.decode('utf-8'))
            metadata_dict = payload.get('metadata', {})
            
            return SerializationMetadata(
                format=SerializationFormat(metadata_dict.get('format', 'json')),
                schema_version=metadata_dict.get('schema_version'),
                compression=CompressionType(metadata_dict.get('compression', 'none')),
                content_type=metadata_dict.get('content_type'),
                size_bytes=metadata_dict.get('size_bytes'),
                metadata=metadata_dict
            )
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            raise SerializationError(str(e))
    
    async def validate_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Validate data against a schema."""
        try:
            # Simple schema validation
            if 'type' in schema:
                expected_type = schema['type']
                if expected_type == 'object' and not isinstance(data, dict):
                    return False
                elif expected_type == 'array' and not isinstance(data, list):
                    return False
                elif expected_type == 'string' and not isinstance(data, str):
                    return False
                elif expected_type == 'number' and not isinstance(data, (int, float)):
                    return False
                elif expected_type == 'boolean' and not isinstance(data, bool):
                    return False
            
            # Validate required fields
            if 'required' in schema and isinstance(data, dict):
                for field in schema['required']:
                    if field not in data:
                        return False
            
            # Validate properties
            if 'properties' in schema and isinstance(data, dict):
                for field, field_schema in schema['properties'].items():
                    if field in data:
                        if not await self.validate_schema(data[field], field_schema):
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    async def register_schema(self, schema_id: str, schema: Dict[str, Any]) -> None:
        """Register a schema for use with this serializer."""
        try:
            self.schemas[schema_id] = schema
            logger.info(f"Registered schema: {schema_id}")
            
        except Exception as e:
            logger.error(f"Schema registration failed: {e}")
            raise SchemaError(str(e))
    
    async def get_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """Get a registered schema."""
        return self.schemas.get(schema_id)
    
    async def get_schema_version(self) -> str:
        """Get the current schema version."""
        return self.current_schema_version
    
    async def set_schema_version(self, version: str) -> None:
        """Set the schema version."""
        try:
            self.current_schema_version = version
            logger.info(f"Schema version set to: {version}")
            
        except Exception as e:
            logger.error(f"Schema version setting failed: {e}")
            raise SchemaError(str(e))
    
    async def compress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using the specified compression type."""
        try:
            if compression_type == CompressionType.GZIP:
                return gzip.compress(data)
            elif compression_type == CompressionType.NONE:
                return data
            else:
                raise SerializationError(f"Unsupported compression type: {compression_type}")
                
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise SerializationError(str(e))
    
    async def decompress(self, compressed_data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using the specified compression type."""
        try:
            if compression_type == CompressionType.GZIP:
                return gzip.decompress(compressed_data)
            elif compression_type == CompressionType.NONE:
                return compressed_data
            else:
                raise SerializationError(f"Unsupported compression type: {compression_type}")
                
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise SerializationError(str(e))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get serialization statistics."""
        return {
            'format': self.get_format().value,
            'content_type': self.get_content_type(),
            'supports_schema_evolution': self.supports_schema_evolution(),
            'supports_compression': self.supports_compression(),
            'supported_compression_types': [ct.value for ct in self.get_supported_compression_types()],
            'current_schema_version': self.current_schema_version,
            'registered_schemas': len(self.schemas),
            'stats': self.stats.copy()
        }
    
    def get_registered_schemas(self) -> List[str]:
        """Get list of registered schema IDs."""
        return list(self.schemas.keys())
    
    def clear_stats(self) -> None:
        """Clear statistics."""
        self.stats = {
            'serializations': 0,
            'deserializations': 0,
            'compressions': 0,
            'decompressions': 0,
            'errors': 0
        }
