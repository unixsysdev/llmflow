"""
TCP Transport Implementation

This module provides a TCP transport implementation with connection pooling
and streaming support for the LLMFlow framework.
"""

import asyncio
import socket
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Set
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager

from ..base import BaseTransport, TransportPlugin, TransportConfig, TransportState
from ...plugins.interfaces.transport import TransportType, TransportError

logger = logging.getLogger(__name__)


class TCPConnectionState(Enum):
    """TCP connection states."""
    CLOSED = "closed"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CLOSING = "closing"
    ERROR = "error"


@dataclass
class TCPConfig(TransportConfig):
    """TCP-specific configuration."""
    connection_pool_size: int = 10
    connection_pool_max_size: int = 100
    connection_idle_timeout: int = 300  # seconds
    tcp_nodelay: bool = True
    tcp_keepalive: bool = True
    tcp_keepalive_idle: int = 60
    tcp_keepalive_interval: int = 60
    tcp_keepalive_count: int = 3
    socket_reuse_address: bool = True
    socket_reuse_port: bool = False
    backlog: int = 128
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'connection_pool_size': self.connection_pool_size,
            'connection_pool_max_size': self.connection_pool_max_size,
            'connection_idle_timeout': self.connection_idle_timeout,
            'tcp_nodelay': self.tcp_nodelay,
            'tcp_keepalive': self.tcp_keepalive,
            'tcp_keepalive_idle': self.tcp_keepalive_idle,
            'tcp_keepalive_interval': self.tcp_keepalive_interval,
            'tcp_keepalive_count': self.tcp_keepalive_count,
            'socket_reuse_address': self.socket_reuse_address,
            'socket_reuse_port': self.socket_reuse_port,
            'backlog': self.backlog
        })
        return base_dict


class TCPConnection:
    """Represents a TCP connection."""
    
    def __init__(self, 
                 reader: asyncio.StreamReader, 
                 writer: asyncio.StreamWriter,
                 remote_address: Tuple[str, int],
                 config: TCPConfig):
        self.reader = reader
        self.writer = writer
        self.remote_address = remote_address
        self.config = config
        self.state = TCPConnectionState.CONNECTED
        self.created_at = asyncio.get_event_loop().time()
        self.last_activity = self.created_at
        self.bytes_sent = 0
        self.bytes_received = 0
        self.messages_sent = 0
        self.messages_received = 0
        self._lock = asyncio.Lock()
    
    async def send(self, data: bytes) -> bool:
        """Send data over the connection."""
        async with self._lock:
            try:
                if self.state != TCPConnectionState.CONNECTED:
                    return False
                
                self.writer.write(data)
                await self.writer.drain()
                
                self.bytes_sent += len(data)
                self.messages_sent += 1
                self.last_activity = asyncio.get_event_loop().time()
                
                return True
                
            except Exception as e:
                logger.error(f"TCP send failed: {e}")
                self.state = TCPConnectionState.ERROR
                return False
    
    async def receive(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Receive data from the connection."""
        async with self._lock:
            try:
                if self.state != TCPConnectionState.CONNECTED:
                    return None
                
                # Read data with timeout
                if timeout:
                    data = await asyncio.wait_for(
                        self.reader.read(self.config.buffer_size),
                        timeout=timeout
                    )
                else:
                    data = await self.reader.read(self.config.buffer_size)
                
                if not data:
                    # Connection closed by peer
                    self.state = TCPConnectionState.CLOSED
                    return None
                
                self.bytes_received += len(data)
                self.messages_received += 1
                self.last_activity = asyncio.get_event_loop().time()
                
                return data
                
            except asyncio.TimeoutError:
                return None
            except Exception as e:
                logger.error(f"TCP receive failed: {e}")
                self.state = TCPConnectionState.ERROR
                return None
    
    async def close(self) -> None:
        """Close the connection."""
        async with self._lock:
            if self.state == TCPConnectionState.CLOSED:
                return
            
            self.state = TCPConnectionState.CLOSING
            
            try:
                if self.writer:
                    self.writer.close()
                    await self.writer.wait_closed()
                    
                self.state = TCPConnectionState.CLOSED
                logger.debug(f"TCP connection to {self.remote_address} closed")
                
            except Exception as e:
                logger.error(f"Error closing TCP connection: {e}")
                self.state = TCPConnectionState.ERROR
    
    def is_idle(self) -> bool:
        """Check if connection is idle."""
        current_time = asyncio.get_event_loop().time()
        return (current_time - self.last_activity) > self.config.connection_idle_timeout
    
    def is_alive(self) -> bool:
        """Check if connection is alive."""
        return self.state == TCPConnectionState.CONNECTED
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'remote_address': self.remote_address,
            'state': self.state.value,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received
        }


class TCPConnectionPool:
    """Connection pool for TCP connections."""
    
    def __init__(self, config: TCPConfig):
        self.config = config
        self.connections: Dict[Tuple[str, int], List[TCPConnection]] = {}
        self.total_connections = 0
        self.connection_stats = {
            'created': 0,
            'reused': 0,
            'closed': 0,
            'errors': 0
        }
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the connection pool."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("TCP connection pool started")
    
    async def stop(self) -> None:
        """Stop the connection pool."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        await self._close_all_connections()
        logger.info("TCP connection pool stopped")
    
    async def get_connection(self, address: str, port: int) -> Optional[TCPConnection]:
        """Get a connection from the pool."""
        async with self._lock:
            endpoint = (address, port)
            
            # Try to reuse existing connection
            if endpoint in self.connections:
                available_connections = self.connections[endpoint]
                
                # Find a healthy connection
                for connection in available_connections[:]:
                    if connection.is_alive():
                        available_connections.remove(connection)
                        self.connection_stats['reused'] += 1
                        return connection
                    else:
                        # Remove dead connection
                        available_connections.remove(connection)
                        self.total_connections -= 1
                        self.connection_stats['closed'] += 1
            
            # Create new connection if possible
            if self.total_connections < self.config.connection_pool_max_size:
                connection = await self._create_connection(address, port)
                if connection:
                    self.total_connections += 1
                    self.connection_stats['created'] += 1
                    return connection
            
            return None
    
    async def return_connection(self, connection: TCPConnection) -> None:
        """Return a connection to the pool."""
        async with self._lock:
            if not connection.is_alive():
                self.total_connections -= 1
                self.connection_stats['closed'] += 1
                return
            
            endpoint = connection.remote_address
            
            if endpoint not in self.connections:
                self.connections[endpoint] = []
            
            # Add to pool if not full
            if len(self.connections[endpoint]) < self.config.connection_pool_size:
                self.connections[endpoint].append(connection)
            else:
                # Pool is full, close the connection
                await connection.close()
                self.total_connections -= 1
                self.connection_stats['closed'] += 1
    
    async def _create_connection(self, address: str, port: int) -> Optional[TCPConnection]:
        """Create a new TCP connection."""
        try:
            reader, writer = await asyncio.open_connection(address, port)
            
            # Configure socket options
            sock = writer.get_extra_info('socket')
            if sock:
                self._configure_socket(sock)
            
            connection = TCPConnection(reader, writer, (address, port), self.config)
            logger.debug(f"Created TCP connection to {address}:{port}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create TCP connection to {address}:{port}: {e}")
            self.connection_stats['errors'] += 1
            return None
    
    def _configure_socket(self, sock: socket.socket) -> None:
        """Configure socket options."""
        try:
            if self.config.tcp_nodelay:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            if self.config.tcp_keepalive:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # Set keepalive parameters (Linux-specific)
                if hasattr(socket, 'TCP_KEEPIDLE'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.config.tcp_keepalive_idle)
                if hasattr(socket, 'TCP_KEEPINTVL'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.config.tcp_keepalive_interval)
                if hasattr(socket, 'TCP_KEEPCNT'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.config.tcp_keepalive_count)
                    
        except Exception as e:
            logger.warning(f"Failed to configure socket options: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background task to cleanup idle connections."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_connections()
            except Exception as e:
                logger.error(f"Error in connection pool cleanup: {e}")
    
    async def _cleanup_idle_connections(self) -> None:
        """Remove idle connections from the pool."""
        async with self._lock:
            endpoints_to_remove = []
            
            for endpoint, connections in self.connections.items():
                idle_connections = []
                
                for connection in connections[:]:
                    if connection.is_idle() or not connection.is_alive():
                        idle_connections.append(connection)
                        connections.remove(connection)
                        self.total_connections -= 1
                        self.connection_stats['closed'] += 1
                
                # Close idle connections
                for connection in idle_connections:
                    await connection.close()
                
                # Remove empty endpoint entries
                if not connections:
                    endpoints_to_remove.append(endpoint)
            
            # Remove empty endpoint entries
            for endpoint in endpoints_to_remove:
                del self.connections[endpoint]
    
    async def _close_all_connections(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            for endpoint, connections in self.connections.items():
                for connection in connections:
                    await connection.close()
                    self.connection_stats['closed'] += 1
            
            self.connections.clear()
            self.total_connections = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'total_connections': self.total_connections,
            'pool_size': len(self.connections),
            'max_pool_size': self.config.connection_pool_max_size,
            'stats': self.connection_stats.copy()
        }


class TCPTransport(BaseTransport):
    """TCP transport implementation with connection pooling."""
    
    def __init__(self, config: TCPConfig):
        super().__init__(config)
        self.connection_pool = TCPConnectionPool(config)
        self.current_connection: Optional[TCPConnection] = None
        self.server_socket: Optional[socket.socket] = None
        self.server_task: Optional[asyncio.Task] = None
        self.client_connections: Set[TCPConnection] = set()
        self.connection_handlers: Dict[Tuple[str, int], asyncio.Task] = {}
    
    def get_transport_type(self) -> TransportType:
        return TransportType.TCP
    
    async def _internal_bind(self) -> bool:
        """Internal bind implementation."""
        try:
            # Start connection pool
            await self.connection_pool.start()
            
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            if self.config.socket_reuse_address:
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            if self.config.socket_reuse_port and hasattr(socket, 'SO_REUSEPORT'):
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            
            self.server_socket.bind((self.config.address, self.config.port))
            self.server_socket.listen(self.config.backlog)
            self.server_socket.setblocking(False)
            
            # Start server task
            self.server_task = asyncio.create_task(self._server_loop())
            
            logger.info(f"TCP transport bound to {self.config.address}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bind TCP transport: {e}")
            await self._cleanup_server()
            return False
    
    async def _internal_connect(self) -> bool:
        """Internal connect implementation."""
        try:
            # Start connection pool
            await self.connection_pool.start()
            
            # Get connection from pool
            self.current_connection = await self.connection_pool.get_connection(
                self.config.address, self.config.port
            )
            
            if self.current_connection:
                logger.info(f"TCP transport connected to {self.config.address}:{self.config.port}")
                return True
            else:
                logger.error(f"Failed to get TCP connection to {self.config.address}:{self.config.port}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect TCP transport: {e}")
            return False
    
    async def _internal_send(self, data: bytes, endpoint: Optional[Tuple[str, int]] = None) -> bool:
        """Internal send implementation."""
        try:
            if endpoint:
                # Send to specific endpoint
                connection = await self.connection_pool.get_connection(endpoint[0], endpoint[1])
                if connection:
                    success = await connection.send(data)
                    await self.connection_pool.return_connection(connection)
                    return success
                return False
            elif self.current_connection:
                # Send on current connection
                return await self.current_connection.send(data)
            else:
                raise TransportError("No connection available")
                
        except Exception as e:
            logger.error(f"TCP send failed: {e}")
            return False
    
    async def _internal_receive(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, Optional[Tuple[str, int]]]]:
        """Internal receive implementation."""
        try:
            if self.current_connection:
                # Receive from current connection
                data = await self.current_connection.receive(timeout)
                if data:
                    return data, self.current_connection.remote_address
                return None
            else:
                # In server mode, this would be handled by connection handlers
                return None
                
        except Exception as e:
            logger.error(f"TCP receive failed: {e}")
            return None
    
    async def _internal_close(self) -> None:
        """Internal close implementation."""
        try:
            # Close current connection
            if self.current_connection:
                await self.connection_pool.return_connection(self.current_connection)
                self.current_connection = None
            
            # Close server
            await self._cleanup_server()
            
            # Stop connection pool
            await self.connection_pool.stop()
            
            logger.info("TCP transport closed")
            
        except Exception as e:
            logger.error(f"TCP close failed: {e}")
    
    async def _server_loop(self) -> None:
        """Server loop for accepting connections."""
        while self.state == TransportState.LISTENING:
            try:
                loop = asyncio.get_event_loop()
                client_socket, client_address = await loop.sock_accept(self.server_socket)
                
                # Configure client socket
                self._configure_client_socket(client_socket)
                
                # Create connection handler
                handler_task = asyncio.create_task(
                    self._handle_client_connection(client_socket, client_address)
                )
                
                self.connection_handlers[client_address] = handler_task
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in server loop: {e}")
                await asyncio.sleep(1)
    
    def _configure_client_socket(self, sock: socket.socket) -> None:
        """Configure client socket options."""
        try:
            if self.config.tcp_nodelay:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            if self.config.tcp_keepalive:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
        except Exception as e:
            logger.warning(f"Failed to configure client socket: {e}")
    
    async def _handle_client_connection(self, client_socket: socket.socket, client_address: Tuple[str, int]) -> None:
        """Handle a client connection."""
        try:
            # Create stream reader/writer
            reader, writer = await asyncio.open_connection(sock=client_socket)
            connection = TCPConnection(reader, writer, client_address, self.config)
            
            self.client_connections.add(connection)
            
            # Emit connection event
            await self._emit_event('client_connected', {'address': client_address})
            
            # Handle connection
            while connection.is_alive():
                data = await connection.receive(timeout=self.config.timeout)
                if data:
                    # Emit data received event
                    await self._emit_event('data_received', {
                        'data': data,
                        'sender': client_address
                    })
                else:
                    break
            
            # Clean up
            self.client_connections.discard(connection)
            await connection.close()
            
            # Emit disconnection event
            await self._emit_event('client_disconnected', {'address': client_address})
            
        except Exception as e:
            logger.error(f"Error handling client connection {client_address}: {e}")
        finally:
            # Remove from connection handlers
            if client_address in self.connection_handlers:
                del self.connection_handlers[client_address]
    
    async def _cleanup_server(self) -> None:
        """Clean up server resources."""
        # Cancel server task
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
            self.server_task = None
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        
        # Cancel connection handlers
        for handler_task in self.connection_handlers.values():
            handler_task.cancel()
        
        # Wait for handlers to finish
        if self.connection_handlers:
            await asyncio.gather(*self.connection_handlers.values(), return_exceptions=True)
        
        self.connection_handlers.clear()
        
        # Close client connections
        for connection in self.client_connections.copy():
            await connection.close()
        
        self.client_connections.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics."""
        base_stats = super().get_stats()
        
        # Add TCP-specific stats
        base_stats.update({
            'connection_pool': self.connection_pool.get_stats(),
            'active_client_connections': len(self.client_connections),
            'active_connection_handlers': len(self.connection_handlers)
        })
        
        return base_stats


class TCPTransportPlugin(TransportPlugin):
    """TCP transport plugin."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def get_name(self) -> str:
        return "tcp_transport"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "TCP transport with connection pooling and streaming support"
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def get_interfaces(self) -> List[Type]:
        from ...plugins.interfaces.transport import ITransportProtocol
        return [ITransportProtocol]
    
    def _create_transport(self, config: TransportConfig) -> BaseTransport:
        tcp_config = TCPConfig(**config.to_dict())
        return TCPTransport(tcp_config)
