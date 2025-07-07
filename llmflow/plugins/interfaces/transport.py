"""
Transport Protocol Interface

This module defines the interface for transport protocols in the LLMFlow framework.
Transport protocols handle the actual network communication between components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class TransportType(Enum):
    """Types of transport protocols."""
    UDP = "udp"
    TCP = "tcp"
    WEBSOCKET = "websocket"
    HTTP = "http"
    GRPC = "grpc"
    QUIC = "quic"


class TransportError(Exception):
    """Base exception for transport-related errors."""
    pass


class ConnectionError(TransportError):
    """Raised when connection operations fail."""
    pass


class SendError(TransportError):
    """Raised when sending messages fails."""
    pass


class ReceiveError(TransportError):
    """Raised when receiving messages fails."""
    pass


class ConnectionInfo:
    """Information about a network connection."""
    
    def __init__(self, 
                 local_address: str, 
                 local_port: int, 
                 remote_address: Optional[str] = None, 
                 remote_port: Optional[int] = None,
                 transport_type: TransportType = TransportType.UDP,
                 metadata: Dict[str, Any] = None):
        self.local_address = local_address
        self.local_port = local_port
        self.remote_address = remote_address
        self.remote_port = remote_port
        self.transport_type = transport_type
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        if self.remote_address:
            return f"{self.transport_type.value}://{self.local_address}:{self.local_port} -> {self.remote_address}:{self.remote_port}"
        else:
            return f"{self.transport_type.value}://{self.local_address}:{self.local_port}"


class ITransportProtocol(ABC):
    """
    Interface for transport protocols in LLMFlow.
    
    This interface defines the contract for all transport implementations,
    including UDP, TCP, WebSocket, HTTP, and other protocols.
    """
    
    @abstractmethod
    def get_transport_type(self) -> TransportType:
        """
        Get the transport type.
        
        Returns:
            The transport type this protocol implements
        """
        pass
    
    @abstractmethod
    async def bind(self, address: str, port: int) -> bool:
        """
        Bind to a local address and port.
        
        Args:
            address: Local address to bind to
            port: Local port to bind to
            
        Returns:
            True if binding was successful, False otherwise
            
        Raises:
            ConnectionError: If binding fails
        """
        pass
    
    @abstractmethod
    async def connect(self, address: str, port: int, timeout: float = 30.0) -> bool:
        """
        Connect to a remote address and port.
        
        Args:
            address: Remote address to connect to
            port: Remote port to connect to
            timeout: Connection timeout in seconds
            
        Returns:
            True if connection was successful, False otherwise
            
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def send(self, data: bytes, endpoint: Optional[Tuple[str, int]] = None) -> bool:
        """
        Send data to the specified endpoint.
        
        Args:
            data: Data to send
            endpoint: Optional endpoint (address, port) for connectionless protocols
            
        Returns:
            True if send was successful, False otherwise
            
        Raises:
            SendError: If sending fails
        """
        pass
    
    @abstractmethod
    async def receive(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, Optional[Tuple[str, int]]]]:
        """
        Receive data from the transport.
        
        Args:
            timeout: Receive timeout in seconds (None for blocking)
            
        Returns:
            Tuple of (data, sender_endpoint) or None if timeout
            
        Raises:
            ReceiveError: If receiving fails
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """
        Close the transport connection.
        
        This method should clean up all resources and close any open connections.
        
        Raises:
            TransportError: If closing fails
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the transport is connected.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def get_connection_info(self) -> Optional[ConnectionInfo]:
        """
        Get information about the current connection.
        
        Returns:
            ConnectionInfo object or None if not connected
        """
        pass
    
    @abstractmethod
    async def set_option(self, option: str, value: Any) -> bool:
        """
        Set a transport-specific option.
        
        Args:
            option: Option name
            value: Option value
            
        Returns:
            True if option was set successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_option(self, option: str) -> Any:
        """
        Get a transport-specific option.
        
        Args:
            option: Option name
            
        Returns:
            Option value or None if not set
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get transport statistics.
        
        Returns:
            Dictionary containing transport statistics
        """
        pass


class TransportPool:
    """
    Pool of transport connections for load balancing and redundancy.
    """
    
    def __init__(self, transport_type: TransportType, pool_size: int = 10):
        self.transport_type = transport_type
        self.pool_size = pool_size
        self.transports: List[ITransportProtocol] = []
        self.current_index = 0
        self._lock = asyncio.Lock()
    
    async def add_transport(self, transport: ITransportProtocol) -> None:
        """
        Add a transport to the pool.
        
        Args:
            transport: Transport to add
        """
        async with self._lock:
            if len(self.transports) < self.pool_size:
                self.transports.append(transport)
            else:
                raise ValueError("Transport pool is full")
    
    async def get_transport(self) -> Optional[ITransportProtocol]:
        """
        Get the next available transport from the pool.
        
        Returns:
            Transport instance or None if pool is empty
        """
        async with self._lock:
            if not self.transports:
                return None
            
            # Round-robin selection
            transport = self.transports[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.transports)
            return transport
    
    async def remove_transport(self, transport: ITransportProtocol) -> bool:
        """
        Remove a transport from the pool.
        
        Args:
            transport: Transport to remove
            
        Returns:
            True if transport was removed, False if not found
        """
        async with self._lock:
            try:
                self.transports.remove(transport)
                # Adjust current index if necessary
                if self.current_index >= len(self.transports) and self.transports:
                    self.current_index = 0
                return True
            except ValueError:
                return False
    
    async def close_all(self) -> None:
        """Close all transports in the pool."""
        async with self._lock:
            for transport in self.transports:
                try:
                    await transport.close()
                except Exception as e:
                    logger.warning(f"Error closing transport: {e}")
            self.transports.clear()
            self.current_index = 0
    
    def size(self) -> int:
        """Get the current pool size."""
        return len(self.transports)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'transport_type': self.transport_type.value,
            'pool_size': self.pool_size,
            'active_transports': len(self.transports),
            'current_index': self.current_index
        }
