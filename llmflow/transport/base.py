"""
Transport Base Module

This module provides the foundation classes for all transport implementations
in the LLMFlow framework.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from ..plugins.interfaces.transport import ITransportProtocol, TransportType, TransportError, ConnectionInfo
from ..plugins.interfaces.base import Plugin

logger = logging.getLogger(__name__)


class TransportState(Enum):
    """Transport connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LISTENING = "listening"
    DISCONNECTING = "disconnecting"
    ERROR = "error"


@dataclass
class TransportConfig:
    """Configuration for transport implementations."""
    address: str = "localhost"
    port: int = 8080
    timeout: float = 30.0
    buffer_size: int = 65536
    max_connections: int = 100
    keep_alive: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    compression: bool = False
    encryption: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'address': self.address,
            'port': self.port,
            'timeout': self.timeout,
            'buffer_size': self.buffer_size,
            'max_connections': self.max_connections,
            'keep_alive': self.keep_alive,
            'retry_attempts': self.retry_attempts,
            'retry_delay': self.retry_delay,
            'compression': self.compression,
            'encryption': self.encryption
        }


@dataclass
class TransportStats:
    """Transport statistics."""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connections_established: int = 0
    connections_failed: int = 0
    errors: int = 0
    last_activity: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'connections_established': self.connections_established,
            'connections_failed': self.connections_failed,
            'errors': self.errors,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }


class BaseTransport(ABC):
    """
    Base class for all transport implementations.
    
    This class provides common functionality and structure for transport protocols.
    """
    
    def __init__(self, config: TransportConfig):
        self.config = config
        self.state = TransportState.DISCONNECTED
        self.stats = TransportStats()
        self.connection_info: Optional[ConnectionInfo] = None
        self.event_handlers: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()
        
    @abstractmethod
    def get_transport_type(self) -> TransportType:
        """Get the transport type."""
        pass
    
    @abstractmethod
    async def _internal_connect(self) -> bool:
        """Internal connection implementation."""
        pass
    
    @abstractmethod
    async def _internal_bind(self) -> bool:
        """Internal bind implementation."""
        pass
    
    @abstractmethod
    async def _internal_send(self, data: bytes, endpoint: Optional[Tuple[str, int]] = None) -> bool:
        """Internal send implementation."""
        pass
    
    @abstractmethod
    async def _internal_receive(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, Optional[Tuple[str, int]]]]:
        """Internal receive implementation."""
        pass
    
    @abstractmethod
    async def _internal_close(self) -> None:
        """Internal close implementation."""
        pass
    
    async def connect(self, address: str, port: int, timeout: float = 30.0) -> bool:
        """Connect to a remote endpoint."""
        async with self._lock:
            try:
                if self.state == TransportState.CONNECTED:
                    logger.warning("Already connected")
                    return True
                
                self.state = TransportState.CONNECTING
                self.config.address = address
                self.config.port = port
                self.config.timeout = timeout
                
                # Emit connecting event
                await self._emit_event('connecting', {'address': address, 'port': port})
                
                # Perform actual connection
                success = await self._internal_connect()
                
                if success:
                    self.state = TransportState.CONNECTED
                    self.stats.connections_established += 1
                    self.connection_info = ConnectionInfo(
                        local_address=self.config.address,
                        local_port=self.config.port,
                        remote_address=address,
                        remote_port=port,
                        transport_type=self.get_transport_type()
                    )
                    
                    # Emit connected event
                    await self._emit_event('connected', {'connection_info': self.connection_info})
                    logger.info(f"Connected to {address}:{port}")
                else:
                    self.state = TransportState.ERROR
                    self.stats.connections_failed += 1
                    await self._emit_event('connection_failed', {'address': address, 'port': port})
                    logger.error(f"Failed to connect to {address}:{port}")
                
                return success
                
            except Exception as e:
                self.state = TransportState.ERROR
                self.stats.connections_failed += 1
                self.stats.errors += 1
                await self._emit_event('error', {'error': str(e), 'operation': 'connect'})
                logger.error(f"Connection error: {e}")
                return False
    
    async def bind(self, address: str, port: int) -> bool:
        """Bind to a local address and port."""
        async with self._lock:
            try:
                if self.state == TransportState.LISTENING:
                    logger.warning("Already listening")
                    return True
                
                self.config.address = address
                self.config.port = port
                
                # Emit binding event
                await self._emit_event('binding', {'address': address, 'port': port})
                
                # Perform actual binding
                success = await self._internal_bind()
                
                if success:
                    self.state = TransportState.LISTENING
                    self.connection_info = ConnectionInfo(
                        local_address=address,
                        local_port=port,
                        transport_type=self.get_transport_type()
                    )
                    
                    # Emit bound event
                    await self._emit_event('bound', {'connection_info': self.connection_info})
                    logger.info(f"Bound to {address}:{port}")
                else:
                    self.state = TransportState.ERROR
                    await self._emit_event('bind_failed', {'address': address, 'port': port})
                    logger.error(f"Failed to bind to {address}:{port}")
                
                return success
                
            except Exception as e:
                self.state = TransportState.ERROR
                self.stats.errors += 1
                await self._emit_event('error', {'error': str(e), 'operation': 'bind'})
                logger.error(f"Bind error: {e}")
                return False
    
    async def send(self, data: bytes, endpoint: Optional[Tuple[str, int]] = None) -> bool:
        """Send data."""
        try:
            if self.state not in [TransportState.CONNECTED, TransportState.LISTENING]:
                raise TransportError("Not connected or listening")
            
            # Emit sending event
            await self._emit_event('sending', {'data_size': len(data), 'endpoint': endpoint})
            
            # Perform actual send
            success = await self._internal_send(data, endpoint)
            
            if success:
                self.stats.messages_sent += 1
                self.stats.bytes_sent += len(data)
                self.stats.last_activity = datetime.utcnow()
                
                # Emit sent event
                await self._emit_event('sent', {'data_size': len(data), 'endpoint': endpoint})
                logger.debug(f"Sent {len(data)} bytes")
            else:
                self.stats.errors += 1
                await self._emit_event('send_failed', {'data_size': len(data), 'endpoint': endpoint})
                logger.error(f"Failed to send {len(data)} bytes")
            
            return success
            
        except Exception as e:
            self.stats.errors += 1
            await self._emit_event('error', {'error': str(e), 'operation': 'send'})
            logger.error(f"Send error: {e}")
            return False
    
    async def receive(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, Optional[Tuple[str, int]]]]:
        """Receive data."""
        try:
            if self.state not in [TransportState.CONNECTED, TransportState.LISTENING]:
                raise TransportError("Not connected or listening")
            
            # Use configured timeout if not specified
            if timeout is None:
                timeout = self.config.timeout
            
            # Emit receiving event
            await self._emit_event('receiving', {'timeout': timeout})
            
            # Perform actual receive
            result = await self._internal_receive(timeout)
            
            if result:
                data, sender = result
                self.stats.messages_received += 1
                self.stats.bytes_received += len(data)
                self.stats.last_activity = datetime.utcnow()
                
                # Emit received event
                await self._emit_event('received', {'data_size': len(data), 'sender': sender})
                logger.debug(f"Received {len(data)} bytes from {sender}")
            
            return result
            
        except Exception as e:
            self.stats.errors += 1
            await self._emit_event('error', {'error': str(e), 'operation': 'receive'})
            logger.error(f"Receive error: {e}")
            return None
    
    async def close(self) -> None:
        """Close the transport."""
        async with self._lock:
            try:
                if self.state == TransportState.DISCONNECTED:
                    return
                
                self.state = TransportState.DISCONNECTING
                
                # Emit closing event
                await self._emit_event('closing', {})
                
                # Perform actual close
                await self._internal_close()
                
                self.state = TransportState.DISCONNECTED
                self.connection_info = None
                
                # Emit closed event
                await self._emit_event('closed', {})
                logger.info("Transport closed")
                
            except Exception as e:
                self.state = TransportState.ERROR
                self.stats.errors += 1
                await self._emit_event('error', {'error': str(e), 'operation': 'close'})
                logger.error(f"Close error: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.state in [TransportState.CONNECTED, TransportState.LISTENING]
    
    def get_connection_info(self) -> Optional[ConnectionInfo]:
        """Get connection information."""
        return self.connection_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics."""
        return {
            'transport_type': self.get_transport_type().value,
            'state': self.state.value,
            'config': self.config.to_dict(),
            'stats': self.stats.to_dict()
        }
    
    def add_event_handler(self, event: str, handler: Callable) -> None:
        """Add an event handler."""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    def remove_event_handler(self, event: str, handler: Callable) -> None:
        """Remove an event handler."""
        if event in self.event_handlers:
            try:
                self.event_handlers[event].remove(handler)
            except ValueError:
                pass
    
    async def _emit_event(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event to all handlers."""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event, data)
                    else:
                        handler(event, data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event}: {e}")
    
    async def set_option(self, option: str, value: Any) -> bool:
        """Set a transport option."""
        try:
            if hasattr(self.config, option):
                setattr(self.config, option, value)
                logger.debug(f"Set option {option} = {value}")
                return True
            else:
                logger.warning(f"Unknown option: {option}")
                return False
        except Exception as e:
            logger.error(f"Error setting option {option}: {e}")
            return False
    
    async def get_option(self, option: str) -> Any:
        """Get a transport option."""
        try:
            if hasattr(self.config, option):
                return getattr(self.config, option)
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting option {option}: {e}")
            return None
    
    def reset_stats(self) -> None:
        """Reset transport statistics."""
        self.stats = TransportStats()
        logger.info("Transport statistics reset")


class TransportPlugin(Plugin, ITransportProtocol):
    """
    Base class for transport plugins.
    
    This class combines the plugin interface with transport functionality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        Plugin.__init__(self, config)
        transport_config = TransportConfig(**config) if config else TransportConfig()
        self.transport = self._create_transport(transport_config)
    
    @abstractmethod
    def _create_transport(self, config: TransportConfig) -> BaseTransport:
        """Create the transport implementation."""
        pass
    
    # Delegate transport methods to the transport instance
    def get_transport_type(self) -> TransportType:
        return self.transport.get_transport_type()
    
    async def bind(self, address: str, port: int) -> bool:
        return await self.transport.bind(address, port)
    
    async def connect(self, address: str, port: int, timeout: float = 30.0) -> bool:
        return await self.transport.connect(address, port, timeout)
    
    async def send(self, data: bytes, endpoint: Optional[Tuple[str, int]] = None) -> bool:
        return await self.transport.send(data, endpoint)
    
    async def receive(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, Optional[Tuple[str, int]]]]:
        return await self.transport.receive(timeout)
    
    async def close(self) -> None:
        return await self.transport.close()
    
    def is_connected(self) -> bool:
        return self.transport.is_connected()
    
    def get_connection_info(self) -> Optional[ConnectionInfo]:
        return self.transport.get_connection_info()
    
    async def set_option(self, option: str, value: Any) -> bool:
        return await self.transport.set_option(option, value)
    
    async def get_option(self, option: str) -> Any:
        return await self.transport.get_option(option)
    
    async def get_stats(self) -> Dict[str, Any]:
        return self.transport.get_stats()
