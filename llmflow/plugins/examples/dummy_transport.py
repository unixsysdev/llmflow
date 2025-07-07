"""
Example Transport Plugin - Dummy UDP Transport

This is a simple example plugin that demonstrates how to implement
the ITransportProtocol interface for LLMFlow.
"""

import asyncio
import socket
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

from ..interfaces.base import Plugin, PluginStatus
from ..interfaces.transport import ITransportProtocol, TransportType, TransportError, ConnectionInfo

logger = logging.getLogger(__name__)


class DummyUDPTransport(Plugin, ITransportProtocol):
    """
    Dummy UDP transport plugin for testing and demonstration.
    
    This plugin implements a basic UDP transport protocol that can be used
    for testing the plugin system without requiring actual network operations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        Plugin.__init__(self, config)
        self.socket: Optional[socket.socket] = None
        self.local_address: Optional[str] = None
        self.local_port: Optional[int] = None
        self.is_bound = False
        self.is_client_connected = False
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'errors': 0
        }
    
    # Plugin interface methods
    def get_name(self) -> str:
        return "dummy_udp_transport"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "Dummy UDP transport plugin for testing and demonstration"
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def get_interfaces(self) -> List[Type]:
        return [ITransportProtocol]
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the transport plugin."""
        self.status = PluginStatus.INITIALIZING
        try:
            self.config.update(config)
            self.status = PluginStatus.INITIALIZED
            logger.info("Dummy UDP transport initialized")
        except Exception as e:
            self.status = PluginStatus.ERROR
            raise e
    
    async def start(self) -> None:
        """Start the transport plugin."""
        self.status = PluginStatus.STARTING
        try:
            # In a real implementation, this would start background tasks
            self.status = PluginStatus.RUNNING
            logger.info("Dummy UDP transport started")
        except Exception as e:
            self.status = PluginStatus.ERROR
            raise e
    
    async def stop(self) -> None:
        """Stop the transport plugin."""
        self.status = PluginStatus.STOPPING
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
            self.is_bound = False
            self.is_client_connected = False
            self.status = PluginStatus.STOPPED
            logger.info("Dummy UDP transport stopped")
        except Exception as e:
            self.status = PluginStatus.ERROR
            raise e
    
    async def shutdown(self) -> None:
        """Shutdown the transport plugin."""
        await self.stop()
        logger.info("Dummy UDP transport shutdown")
    
    async def health_check(self) -> bool:
        """Check if the transport is healthy."""
        return self.status == PluginStatus.RUNNING
    
    # Transport interface methods
    def get_transport_type(self) -> TransportType:
        return TransportType.UDP
    
    async def bind(self, address: str, port: int) -> bool:
        """Bind to a local address and port."""
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to address and port
            self.socket.bind((address, port))
            self.local_address = address
            self.local_port = port
            self.is_bound = True
            
            logger.info(f"Dummy UDP transport bound to {address}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bind UDP transport: {e}")
            self.stats['errors'] += 1
            return False
    
    async def connect(self, address: str, port: int, timeout: float = 30.0) -> bool:
        """Connect to a remote address and port."""
        try:
            if not self.socket:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # For UDP, we just store the remote address
            self.remote_address = address
            self.remote_port = port
            self.is_client_connected = True
            
            logger.info(f"Dummy UDP transport connected to {address}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect UDP transport: {e}")
            self.stats['errors'] += 1
            return False
    
    async def send(self, data: bytes, endpoint: Optional[Tuple[str, int]] = None) -> bool:
        """Send data to the specified endpoint."""
        try:
            if not self.socket:
                raise TransportError("Socket not created")
            
            # Use provided endpoint or default remote endpoint
            if endpoint:
                target_address, target_port = endpoint
            elif hasattr(self, 'remote_address') and hasattr(self, 'remote_port'):
                target_address, target_port = self.remote_address, self.remote_port
            else:
                raise TransportError("No endpoint specified and no default remote endpoint")
            
            # Send data
            bytes_sent = self.socket.sendto(data, (target_address, target_port))
            
            # Update stats
            self.stats['messages_sent'] += 1
            self.stats['bytes_sent'] += bytes_sent
            
            logger.debug(f"Sent {bytes_sent} bytes to {target_address}:{target_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            self.stats['errors'] += 1
            return False
    
    async def receive(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, Optional[Tuple[str, int]]]]:
        """Receive data from the transport."""
        try:
            if not self.socket:
                raise TransportError("Socket not created")
            
            # Set socket timeout
            if timeout:
                self.socket.settimeout(timeout)
            
            # Receive data
            data, address = self.socket.recvfrom(self.config.get('buffer_size', 65536))
            
            # Update stats
            self.stats['messages_received'] += 1
            self.stats['bytes_received'] += len(data)
            
            logger.debug(f"Received {len(data)} bytes from {address}")
            return data, address
            
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Failed to receive data: {e}")
            self.stats['errors'] += 1
            return None
    
    async def close(self) -> None:
        """Close the transport connection."""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.is_bound = False
        self.is_client_connected = False
        logger.info("Dummy UDP transport closed")
    
    def is_connected(self) -> bool:
        """Check if the transport is connected."""
        return self.is_bound or self.is_client_connected
    
    def get_connection_info(self) -> Optional[ConnectionInfo]:
        """Get information about the current connection."""
        if not self.is_connected():
            return None
        
        return ConnectionInfo(
            local_address=self.local_address or "0.0.0.0",
            local_port=self.local_port or 0,
            remote_address=getattr(self, 'remote_address', None),
            remote_port=getattr(self, 'remote_port', None),
            transport_type=TransportType.UDP
        )
    
    async def set_option(self, option: str, value: Any) -> bool:
        """Set a transport-specific option."""
        try:
            if option == 'buffer_size':
                self.config['buffer_size'] = int(value)
                return True
            elif option == 'timeout':
                self.config['timeout'] = float(value)
                return True
            else:
                logger.warning(f"Unknown option: {option}")
                return False
        except Exception as e:
            logger.error(f"Failed to set option {option}: {e}")
            return False
    
    async def get_option(self, option: str) -> Any:
        """Get a transport-specific option."""
        return self.config.get(option)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics."""
        return {
            'transport_type': self.get_transport_type().value,
            'is_connected': self.is_connected(),
            'local_address': self.local_address,
            'local_port': self.local_port,
            'remote_address': getattr(self, 'remote_address', None),
            'remote_port': getattr(self, 'remote_port', None),
            'stats': self.stats.copy()
        }
