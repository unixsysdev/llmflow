"""
Storage Provider Interface

This module defines the interface for storage providers in the LLMFlow framework.
Storage providers handle persistent storage of data, messages, and metadata.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """Types of storage providers."""
    IN_MEMORY = "in_memory"
    FILE_SYSTEM = "file_system"
    REDIS = "redis"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    CASSANDRA = "cassandra"
    ELASTICSEARCH = "elasticsearch"
    S3 = "s3"
    CUSTOM = "custom"


class ConsistencyLevel(Enum):
    """Consistency levels for storage operations."""
    EVENTUAL = "eventual"
    STRONG = "strong"
    CAUSAL = "causal"


class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass


class ConnectionError(StorageError):
    """Raised when storage connection fails."""
    pass


class WriteError(StorageError):
    """Raised when write operations fail."""
    pass


class ReadError(StorageError):
    """Raised when read operations fail."""
    pass


class DeleteError(StorageError):
    """Raised when delete operations fail."""
    pass


class StorageRecord:
    """
    Represents a record in storage.
    """
    
    def __init__(self, 
                 key: str, 
                 data: Any,
                 metadata: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[datetime] = None,
                 version: Optional[str] = None):
        self.key = key
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.utcnow()
        self.version = version
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            'key': self.key,
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version
        }


class StorageQuery:
    """
    Represents a query for storage operations.
    """
    
    def __init__(self, 
                 filters: Optional[Dict[str, Any]] = None,
                 sort: Optional[List[Tuple[str, str]]] = None,
                 limit: Optional[int] = None,
                 offset: Optional[int] = None):
        self.filters = filters or {}
        self.sort = sort or []
        self.limit = limit
        self.offset = offset
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary."""
        return {
            'filters': self.filters,
            'sort': self.sort,
            'limit': self.limit,
            'offset': self.offset
        }


class IStorageProvider(ABC):
    """
    Interface for storage providers in LLMFlow.
    
    This interface defines the contract for all storage implementations,
    including in-memory, file system, database, and cloud storage.
    """
    
    @abstractmethod
    def get_storage_type(self) -> StorageType:
        """
        Get the storage type.
        
        Returns:
            The storage type this provider implements
        """
        pass
    
    @abstractmethod
    def get_consistency_level(self) -> ConsistencyLevel:
        """
        Get the consistency level.
        
        Returns:
            The consistency level this provider guarantees
        """
        pass
    
    @abstractmethod
    async def connect(self, connection_string: str, **kwargs) -> None:
        """
        Connect to the storage system.
        
        Args:
            connection_string: Connection string for the storage system
            **kwargs: Additional connection parameters
            
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the storage system.
        
        Raises:
            StorageError: If disconnection fails
        """
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if connected to the storage system.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    async def put(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store data with the given key.
        
        Args:
            key: Storage key
            data: Data to store
            metadata: Optional metadata
            
        Returns:
            True if storage was successful, False otherwise
            
        Raises:
            WriteError: If storage fails
        """
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[StorageRecord]:
        """
        Retrieve data by key.
        
        Args:
            key: Storage key
            
        Returns:
            StorageRecord if found, None otherwise
            
        Raises:
            ReadError: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete data by key.
        
        Args:
            key: Storage key
            
        Returns:
            True if deletion was successful, False otherwise
            
        Raises:
            DeleteError: If deletion fails
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in storage.
        
        Args:
            key: Storage key
            
        Returns:
            True if key exists, False otherwise
            
        Raises:
            ReadError: If check fails
        """
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: Optional[str] = None) -> AsyncIterator[str]:
        """
        List all keys in storage.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Yields:
            Storage keys
            
        Raises:
            ReadError: If listing fails
        """
        pass
    
    @abstractmethod
    async def query(self, query: StorageQuery) -> AsyncIterator[StorageRecord]:
        """
        Query storage for records matching the criteria.
        
        Args:
            query: Query criteria
            
        Yields:
            Storage records matching the query
            
        Raises:
            ReadError: If query fails
        """
        pass
    
    @abstractmethod
    async def count(self, query: Optional[StorageQuery] = None) -> int:
        """
        Count records matching the query.
        
        Args:
            query: Optional query criteria
            
        Returns:
            Number of matching records
            
        Raises:
            ReadError: If count fails
        """
        pass
    
    @abstractmethod
    async def batch_put(self, records: List[Tuple[str, Any, Optional[Dict[str, Any]]]]) -> List[bool]:
        """
        Store multiple records in a batch.
        
        Args:
            records: List of (key, data, metadata) tuples
            
        Returns:
            List of success/failure indicators
            
        Raises:
            WriteError: If batch operation fails
        """
        pass
    
    @abstractmethod
    async def batch_get(self, keys: List[str]) -> List[Optional[StorageRecord]]:
        """
        Retrieve multiple records by keys.
        
        Args:
            keys: List of storage keys
            
        Returns:
            List of StorageRecord objects (None for missing keys)
            
        Raises:
            ReadError: If batch retrieval fails
        """
        pass
    
    @abstractmethod
    async def batch_delete(self, keys: List[str]) -> List[bool]:
        """
        Delete multiple records by keys.
        
        Args:
            keys: List of storage keys
            
        Returns:
            List of success/failure indicators
            
        Raises:
            DeleteError: If batch deletion fails
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all data from storage.
        
        Returns:
            True if clear was successful, False otherwise
            
        Raises:
            StorageError: If clear fails
        """
        pass
    
    @abstractmethod
    async def backup(self, backup_path: str) -> bool:
        """
        Create a backup of the storage.
        
        Args:
            backup_path: Path for the backup
            
        Returns:
            True if backup was successful, False otherwise
            
        Raises:
            StorageError: If backup fails
        """
        pass
    
    @abstractmethod
    async def restore(self, backup_path: str) -> bool:
        """
        Restore storage from a backup.
        
        Args:
            backup_path: Path to the backup
            
        Returns:
            True if restore was successful, False otherwise
            
        Raises:
            StorageError: If restore fails
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check the health of the storage system.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def transaction_begin(self) -> str:
        """
        Begin a transaction (if supported).
        
        Returns:
            Transaction ID
            
        Raises:
            StorageError: If transactions are not supported or begin fails
        """
        pass
    
    @abstractmethod
    async def transaction_commit(self, transaction_id: str) -> bool:
        """
        Commit a transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            True if commit was successful, False otherwise
            
        Raises:
            StorageError: If commit fails
        """
        pass
    
    @abstractmethod
    async def transaction_rollback(self, transaction_id: str) -> bool:
        """
        Rollback a transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            True if rollback was successful, False otherwise
            
        Raises:
            StorageError: If rollback fails
        """
        pass
    
    @abstractmethod
    async def create_index(self, field: str, index_type: str = "btree") -> bool:
        """
        Create an index on a field.
        
        Args:
            field: Field to index
            index_type: Type of index
            
        Returns:
            True if index creation was successful, False otherwise
            
        Raises:
            StorageError: If index creation fails
        """
        pass
    
    @abstractmethod
    async def drop_index(self, field: str) -> bool:
        """
        Drop an index on a field.
        
        Args:
            field: Field to drop index from
            
        Returns:
            True if index drop was successful, False otherwise
            
        Raises:
            StorageError: If index drop fails
        """
        pass
    
    @abstractmethod
    async def list_indexes(self) -> List[str]:
        """
        List all indexes.
        
        Returns:
            List of indexed fields
            
        Raises:
            StorageError: If listing fails
        """
        pass
