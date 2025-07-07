"""
WebSocket Transport Implementation

This module provides a WebSocket transport implementation for browser compatibility
and real-time communication in the LLMFlow framework.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from dataclasses import dataclass
from enum import Enum
import base64
import hashlib
import struct
import socket

from ..base import BaseTransport, TransportPlugin, TransportConfig
from ...plugins.interfaces.transport import TransportType, TransportError

logger = logging.getLogger(__name__)

# WebSocket protocol constants
WS_MAGIC_STRING = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
WS_VERSION = 13


class WebSocketOpcode(Enum):
    """WebSocket frame opcodes."""
    CONTINUATION = 0x0
    TEXT = 0x1
    BINARY = 0x2
    CLOSE = 0x8
    PING = 0x9
    PONG = 0xa


class WebSocketState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class WebSocketConfig(TransportConfig):
    """WebSocket-specific configuration."""
    max_frame_size: int = 1024 * 1024  # 1MB
    ping_interval: float = 30.0  # seconds
    ping_timeout: float = 10.0  # seconds
    close_timeout: float = 10.0  # seconds
    auto_ping: bool = True
    compression: bool = False
    subprotocols: List[str] = None
    backlog: int = 128
    
    def __post_init__(self):
        if self.subprotocols is None:
            self.subprotocols = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'max_frame_size': self.max_frame_size,
            'ping_interval': self.ping_interval,
            'ping_timeout': self.ping_timeout,
            'close_timeout': self.close_timeout,
            'auto_ping': self.auto_ping,
            'compression': self.compression,
            'subprotocols': self.subprotocols,
            'backlog': self.backlog
        })
        return base_dict


@dataclass
class WebSocketFrame:
    """WebSocket frame structure."""
    fin: bool
    opcode: WebSocketOpcode
    masked: bool
    payload: bytes
    mask: Optional[bytes] = None
    
    def to_bytes(self) -> bytes:
        """Serialize frame to bytes."""
        # First byte: FIN + RSV + Opcode
        byte1 = (0x80 if self.fin else 0x00) | self.opcode.value
        
        # Second byte: MASK + Payload length
        payload_len = len(self.payload)
        
        if payload_len < 126:
            byte2 = (0x80 if self.masked else 0x00) | payload_len
            length_bytes = struct.pack('!BB', byte1, byte2)
        elif payload_len < 65536:
            byte2 = (0x80 if self.masked else 0x00) | 126
            length_bytes = struct.pack('!BBH', byte1, byte2, payload_len)
        else:
            byte2 = (0x80 if self.masked else 0x00) | 127
            length_bytes = struct.pack('!BBQ', byte1, byte2, payload_len)
        
        frame_data = length_bytes
        
        # Add mask if present
        if self.masked and self.mask:
            frame_data += self.mask
        
        # Add payload (masked if necessary)
        if self.masked and self.mask:
            masked_payload = bytearray()
            for i, byte in enumerate(self.payload):
                masked_payload.append(byte ^ self.mask[i % 4])
            frame_data += bytes(masked_payload)
        else:
            frame_data += self.payload
        
        return frame_data
    
    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple['WebSocketFrame', int]:
        """Parse frame from bytes."""
        if len(data) < offset + 2:
            raise ValueError("Insufficient data for WebSocket frame")
        
        # Parse first two bytes
        byte1 = data[offset]
        byte2 = data[offset + 1]
        
        fin = bool(byte1 & 0x80)
        opcode = WebSocketOpcode(byte1 & 0x0f)
        masked = bool(byte2 & 0x80)
        payload_len = byte2 & 0x7f
        
        bytes_consumed = 2
        
        # Parse extended payload length
        if payload_len == 126:
            if len(data) < offset + 4:
                raise ValueError("Insufficient data for extended payload length")
            payload_len = struct.unpack('!H', data[offset + 2:offset + 4])[0]
            bytes_consumed += 2
        elif payload_len == 127:
            if len(data) < offset + 10:
                raise ValueError("Insufficient data for extended payload length")
            payload_len = struct.unpack('!Q', data[offset + 2:offset + 10])[0]
            bytes_consumed += 8
        
        # Parse mask
        mask = None
        if masked:
            if len(data) < offset + bytes_consumed + 4:
                raise ValueError("Insufficient data for mask")
            mask = data[offset + bytes_consumed:offset + bytes_consumed + 4]
            bytes_consumed += 4
        
        # Parse payload
        if len(data) < offset + bytes_consumed + payload_len:
            raise ValueError("Insufficient data for payload")
        
        payload = data[offset + bytes_consumed:offset + bytes_consumed + payload_len]
        
        # Unmask payload if necessary
        if masked and mask:
            unmasked_payload = bytearray()
            for i, byte in enumerate(payload):
                unmasked_payload.append(byte ^ mask[i % 4])
            payload = bytes(unmasked_payload)
        
        bytes_consumed += payload_len
        
        return cls(fin=fin, opcode=opcode, masked=masked, payload=payload, mask=mask), bytes_consumed


class WebSocketConnection:
    """WebSocket connection handler."""
    
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, config: WebSocketConfig):
        self.reader = reader
        self.writer = writer
        self.config = config
        self.state = WebSocketState.CONNECTING
        self.receive_buffer = bytearray()
        self.ping_task: Optional[asyncio.Task] = None
        self.pong_received = asyncio.Event()
        self.close_code: Optional[int] = None
        self.close_reason: Optional[str] = None
        self.stats = {
            'frames_sent': 0,
            'frames_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'pings_sent': 0,
            'pongs_received': 0
        }
    
    async def send_frame(self, frame: WebSocketFrame) -> bool:
        """Send a WebSocket frame."""
        try:
            if self.state != WebSocketState.OPEN:
                return False
            
            frame_data = frame.to_bytes()
            self.writer.write(frame_data)
            await self.writer.drain()
            
            self.stats['frames_sent'] += 1
            self.stats['bytes_sent'] += len(frame_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WebSocket frame: {e}")
            return False
    
    async def receive_frame(self, timeout: Optional[float] = None) -> Optional[WebSocketFrame]:
        """Receive a WebSocket frame."""
        try:
            if self.state != WebSocketState.OPEN:
                return None
            
            # Read data into buffer
            if timeout:
                data = await asyncio.wait_for(
                    self.reader.read(self.config.buffer_size),
                    timeout=timeout
                )
            else:
                data = await self.reader.read(self.config.buffer_size)
            
            if not data:
                return None
            
            self.receive_buffer.extend(data)
            
            # Try to parse frame
            try:
                frame, bytes_consumed = WebSocketFrame.from_bytes(self.receive_buffer)
                
                # Remove consumed bytes from buffer
                self.receive_buffer = self.receive_buffer[bytes_consumed:]
                
                self.stats['frames_received'] += 1
                self.stats['bytes_received'] += bytes_consumed
                
                return frame
                
            except ValueError:
                # Not enough data yet
                return None
                
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Failed to receive WebSocket frame: {e}")
            return None
    
    async def send_binary(self, data: bytes) -> bool:
        """Send binary message."""
        frame = WebSocketFrame(
            fin=True,
            opcode=WebSocketOpcode.BINARY,
            masked=False,
            payload=data
        )
        return await self.send_frame(frame)
    
    async def send_ping(self, data: bytes = b'') -> bool:
        """Send ping frame."""
        frame = WebSocketFrame(
            fin=True,
            opcode=WebSocketOpcode.PING,
            masked=False,
            payload=data
        )
        
        success = await self.send_frame(frame)
        if success:
            self.stats['pings_sent'] += 1
        
        return success
    
    async def send_pong(self, data: bytes = b'') -> bool:
        """Send pong frame."""
        frame = WebSocketFrame(
            fin=True,
            opcode=WebSocketOpcode.PONG,
            masked=False,
            payload=data
        )
        return await self.send_frame(frame)
    
    async def send_close(self, code: int = 1000, reason: str = '') -> bool:
        """Send close frame."""
        payload = struct.pack('!H', code) + reason.encode('utf-8')
        frame = WebSocketFrame(
            fin=True,
            opcode=WebSocketOpcode.CLOSE,
            masked=False,
            payload=payload
        )
        
        self.state = WebSocketState.CLOSING
        return await self.send_frame(frame)
    
    async def handle_frame(self, frame: WebSocketFrame) -> Optional[bytes]:
        """Handle received frame."""
        if frame.opcode == WebSocketOpcode.TEXT:
            return frame.payload
        elif frame.opcode == WebSocketOpcode.BINARY:
            return frame.payload
        elif frame.opcode == WebSocketOpcode.PING:
            # Send pong response
            await self.send_pong(frame.payload)
            return None
        elif frame.opcode == WebSocketOpcode.PONG:
            # Handle pong response
            self.pong_received.set()
            self.stats['pongs_received'] += 1
            return None
        elif frame.opcode == WebSocketOpcode.CLOSE:
            # Handle close frame
            if len(frame.payload) >= 2:
                self.close_code = struct.unpack('!H', frame.payload[:2])[0]
                self.close_reason = frame.payload[2:].decode('utf-8')
            
            self.state = WebSocketState.CLOSED
            return None
        
        return None
    
    async def close(self, code: int = 1000, reason: str = '') -> None:
        """Close the WebSocket connection."""
        if self.state == WebSocketState.CLOSED:
            return
        
        if self.state == WebSocketState.OPEN:
            await self.send_close(code, reason)
        
        # Cancel ping task
        if self.ping_task:
            self.ping_task.cancel()
        
        # Close the underlying connection
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception as e:
            logger.debug(f"Error closing WebSocket writer: {e}")
        
        self.state = WebSocketState.CLOSED
    
    async def start_ping_task(self) -> None:
        """Start the ping task."""
        if self.config.auto_ping and self.ping_task is None:
            self.ping_task = asyncio.create_task(self._ping_loop())
    
    async def _ping_loop(self) -> None:
        """Ping loop to keep connection alive."""
        while self.state == WebSocketState.OPEN:
            try:
                await asyncio.sleep(self.config.ping_interval)
                
                if self.state != WebSocketState.OPEN:
                    break
                
                # Send ping
                await self.send_ping()
                
                # Wait for pong
                self.pong_received.clear()
                try:
                    await asyncio.wait_for(
                        self.pong_received.wait(),
                        timeout=self.config.ping_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Ping timeout, closing connection")
                    await self.close(code=1001, reason="Ping timeout")
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping loop: {e}")
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return self.stats.copy()


class WebSocketTransport(BaseTransport):
    """WebSocket transport implementation."""
    
    def __init__(self, config: WebSocketConfig):
        super().__init__(config)
        self.connection: Optional[WebSocketConnection] = None
        self.server_socket: Optional[socket.socket] = None
        self.server_task: Optional[asyncio.Task] = None
        self.client_connections: Dict[str, WebSocketConnection] = {}
    
    def get_transport_type(self) -> TransportType:
        return TransportType.WEBSOCKET
    
    async def _internal_bind(self) -> bool:
        """Internal bind implementation."""
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.config.address, self.config.port))
            self.server_socket.listen(self.config.backlog)
            self.server_socket.setblocking(False)
            
            # Start server task
            self.server_task = asyncio.create_task(self._server_loop())
            
            logger.info(f"WebSocket transport bound to {self.config.address}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bind WebSocket transport: {e}")
            return False
    
    async def _internal_connect(self) -> bool:
        """Internal connect implementation."""
        try:
            # Create TCP connection
            reader, writer = await asyncio.open_connection(
                self.config.address, self.config.port
            )
            
            # Perform WebSocket handshake
            success = await self._perform_client_handshake(reader, writer)
            
            if success:
                self.connection = WebSocketConnection(reader, writer, self.config)
                self.connection.state = WebSocketState.OPEN
                await self.connection.start_ping_task()
                
                logger.info(f"WebSocket transport connected to {self.config.address}:{self.config.port}")
                return True
            else:
                writer.close()
                await writer.wait_closed()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect WebSocket transport: {e}")
            return False
    
    async def _internal_send(self, data: bytes, endpoint: Optional[Tuple[str, int]] = None) -> bool:
        """Internal send implementation."""
        try:
            if self.connection:
                return await self.connection.send_binary(data)
            else:
                raise TransportError("No WebSocket connection available")
                
        except Exception as e:
            logger.error(f"WebSocket send failed: {e}")
            return False
    
    async def _internal_receive(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, Optional[Tuple[str, int]]]]:
        """Internal receive implementation."""
        try:
            if self.connection:
                frame = await self.connection.receive_frame(timeout)
                if frame:
                    data = await self.connection.handle_frame(frame)
                    if data:
                        return data, None
            return None
            
        except Exception as e:
            logger.error(f"WebSocket receive failed: {e}")
            return None
    
    async def _internal_close(self) -> None:
        """Internal close implementation."""
        try:
            # Close client connection
            if self.connection:
                await self.connection.close()
                self.connection = None
            
            # Close server
            if self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
                self.server_task = None
            
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            
            # Close client connections
            for connection in self.client_connections.values():
                await connection.close()
            self.client_connections.clear()
            
            logger.info("WebSocket transport closed")
            
        except Exception as e:
            logger.error(f"WebSocket close failed: {e}")
    
    async def _perform_client_handshake(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
        """Perform WebSocket client handshake."""
        try:
            # Generate WebSocket key
            import os
            websocket_key = base64.b64encode(os.urandom(16)).decode('ascii')
            
            # Send handshake request
            handshake_request = (
                f"GET / HTTP/1.1\r\n"
                f"Host: {self.config.address}:{self.config.port}\r\n"
                f"Upgrade: websocket\r\n"
                f"Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {websocket_key}\r\n"
                f"Sec-WebSocket-Version: {WS_VERSION}\r\n"
                f"\r\n"
            )
            
            writer.write(handshake_request.encode('utf-8'))
            await writer.drain()
            
            # Read response
            response = await reader.read(1024)
            response_str = response.decode('utf-8')
            
            # Verify handshake response
            if "HTTP/1.1 101" not in response_str:
                logger.error("WebSocket handshake failed: Invalid response")
                return False
            
            # Verify Sec-WebSocket-Accept
            expected_accept = base64.b64encode(
                hashlib.sha1((websocket_key + WS_MAGIC_STRING).encode('utf-8')).digest()
            ).decode('ascii')
            
            if f"Sec-WebSocket-Accept: {expected_accept}" not in response_str:
                logger.error("WebSocket handshake failed: Invalid Sec-WebSocket-Accept")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket handshake failed: {e}")
            return False
    
    async def _server_loop(self) -> None:
        """Server loop for accepting WebSocket connections."""
        while True:
            try:
                loop = asyncio.get_event_loop()
                client_socket, client_address = await loop.sock_accept(self.server_socket)
                
                # Handle client connection
                asyncio.create_task(self._handle_client_connection(client_socket, client_address))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket server loop: {e}")
                await asyncio.sleep(1)
    
    async def _handle_client_connection(self, client_socket: socket.socket, client_address: Tuple[str, int]) -> None:
        """Handle a client WebSocket connection."""
        try:
            # Create stream reader/writer
            reader, writer = await asyncio.open_connection(sock=client_socket)
            
            # Perform server handshake
            success = await self._perform_server_handshake(reader, writer)
            
            if success:
                # Create WebSocket connection
                connection = WebSocketConnection(reader, writer, self.config)
                connection.state = WebSocketState.OPEN
                
                connection_id = f"{client_address[0]}:{client_address[1]}"
                self.client_connections[connection_id] = connection
                
                await connection.start_ping_task()
                
                # Emit connection event
                await self._emit_event('websocket_connected', {'address': client_address})
                
                # Handle frames
                while connection.state == WebSocketState.OPEN:
                    frame = await connection.receive_frame(timeout=self.config.timeout)
                    if frame:
                        data = await connection.handle_frame(frame)
                        if data:
                            # Emit data received event
                            await self._emit_event('websocket_data_received', {
                                'data': data,
                                'sender': client_address
                            })
                    else:
                        break
                
                # Clean up
                del self.client_connections[connection_id]
                await connection.close()
                
                # Emit disconnection event
                await self._emit_event('websocket_disconnected', {'address': client_address})
            else:
                writer.close()
                await writer.wait_closed()
                
        except Exception as e:
            logger.error(f"Error handling WebSocket client connection {client_address}: {e}")
    
    async def _perform_server_handshake(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
        """Perform WebSocket server handshake."""
        try:
            # Read HTTP request
            request_data = await reader.read(1024)
            request_str = request_data.decode('utf-8')
            
            # Parse request
            lines = request_str.split('\r\n')
            headers = {}
            
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Validate WebSocket request
            if (headers.get('upgrade', '').lower() != 'websocket' or
                headers.get('connection', '').lower() != 'upgrade' or
                headers.get('sec-websocket-version') != str(WS_VERSION)):
                
                # Send 400 Bad Request
                response = (
                    "HTTP/1.1 400 Bad Request\r\n"
                    "Content-Length: 0\r\n"
                    "\r\n"
                )
                writer.write(response.encode('utf-8'))
                await writer.drain()
                return False
            
            # Calculate Sec-WebSocket-Accept
            websocket_key = headers.get('sec-websocket-key', '')
            if not websocket_key:
                return False
            
            accept_key = base64.b64encode(
                hashlib.sha1((websocket_key + WS_MAGIC_STRING).encode('utf-8')).digest()
            ).decode('ascii')
            
            # Send handshake response
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_key}\r\n"
                "\r\n"
            )
            
            writer.write(response.encode('utf-8'))
            await writer.drain()
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket server handshake failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics."""
        base_stats = super().get_stats()
        
        # Add WebSocket-specific stats
        base_stats.update({
            'client_connections': len(self.client_connections),
            'connection_stats': {
                conn_id: conn.get_stats() 
                for conn_id, conn in self.client_connections.items()
            }
        })
        
        if self.connection:
            base_stats['main_connection_stats'] = self.connection.get_stats()
        
        return base_stats


class WebSocketTransportPlugin(TransportPlugin):
    """WebSocket transport plugin."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def get_name(self) -> str:
        return "websocket_transport"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "WebSocket transport for browser compatibility and real-time communication"
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def get_interfaces(self) -> List[Type]:
        from ...plugins.interfaces.transport import ITransportProtocol
        return [ITransportProtocol]
    
    def _create_transport(self, config: TransportConfig) -> BaseTransport:
        ws_config = WebSocketConfig(**config.to_dict())
        return WebSocketTransport(ws_config)
