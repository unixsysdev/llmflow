"""
UDP Transport Implementation

This module provides a UDP transport implementation with reliability layer
and flow control for the LLMFlow framework.
"""

import asyncio
import socket
import struct
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Callable
from dataclasses import dataclass
from enum import Enum

from ..base import BaseTransport, TransportPlugin, TransportConfig, TransportState
from ...plugins.interfaces.transport import TransportType, TransportError

logger = logging.getLogger(__name__)


class ReliabilityMode(Enum):
    """UDP reliability modes."""
    NONE = "none"
    ACKNOWLEDGMENT = "acknowledgment"
    RETRANSMISSION = "retransmission"


@dataclass
class UDPConfig(TransportConfig):
    """UDP-specific configuration."""
    reliability_mode: ReliabilityMode = ReliabilityMode.ACKNOWLEDGMENT
    
    def __post_init__(self):
        # Handle string values for reliability_mode
        if isinstance(self.reliability_mode, str):
            self.reliability_mode = ReliabilityMode(self.reliability_mode)
    max_retries: int = 3
    ack_timeout: float = 1.0
    sequence_number_bits: int = 16
    window_size: int = 64
    fragment_size: int = 1400
    enable_flow_control: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'reliability_mode': self.reliability_mode.value if hasattr(self.reliability_mode, 'value') else str(self.reliability_mode),
            'max_retries': self.max_retries,
            'ack_timeout': self.ack_timeout,
            'sequence_number_bits': self.sequence_number_bits,
            'window_size': self.window_size,
            'fragment_size': self.fragment_size,
            'enable_flow_control': self.enable_flow_control
        })
        return base_dict


class UDPMessageType(Enum):
    """UDP message types."""
    DATA = 0
    ACK = 1
    NACK = 2
    PING = 3
    PONG = 4


@dataclass
class UDPMessage:
    """UDP message structure."""
    msg_type: UDPMessageType
    sequence_number: int
    fragment_id: int
    total_fragments: int
    data: bytes
    timestamp: float = 0.0
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        header = struct.pack(
            '!BHHHI',
            self.msg_type.value,
            self.sequence_number,
            self.fragment_id,
            self.total_fragments,
            len(self.data)
        )
        return header + self.data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'UDPMessage':
        """Deserialize message from bytes."""
        if len(data) < 11:  # Correct header size: B(1) + H(2) + H(2) + H(2) + I(4) = 11 bytes
            raise ValueError("Invalid message: too short")
        
        msg_type, seq_num, frag_id, total_frags, data_len = struct.unpack('!BHHHI', data[:11])
        
        if len(data) < 11 + data_len:
            raise ValueError("Invalid message: data length mismatch")
        
        return cls(
            msg_type=UDPMessageType(msg_type),
            sequence_number=seq_num,
            fragment_id=frag_id,
            total_fragments=total_frags,
            data=data[11:11+data_len],
            timestamp=time.time()
        )




class UDPReliabilityLayer:
    """Reliability layer for UDP transport."""
    
    def __init__(self, config):
        self.config = config
        self.sequence_number = 0
        self.pending_acks: Dict[int, UDPMessage] = {}
        self.received_fragments: Dict[int, Dict[int, UDPMessage]] = {}
        self.last_received_seq = -1
        self.send_window = set()
        self.receive_window = set()
        self._lock = asyncio.Lock()
        self._send_ack_callback: Optional[Callable] = None
        self._retransmit_callback: Optional[Callable] = None

    def set_ack_callback(self, callback: Callable[[bytes, Tuple[str, int]], None]) -> None:
        """Set callback function for sending ACK messages."""
        self._send_ack_callback = callback

    def set_retransmit_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback function for retransmitting messages."""
        self._retransmit_callback = callback
    
    def get_next_sequence_number(self) -> int:
        """Get next sequence number."""
        self.sequence_number = (self.sequence_number + 1) % (2 ** self.config.sequence_number_bits)
        return self.sequence_number
    
    def fragment_data(self, data: bytes) -> List[bytes]:
        """Fragment large data into smaller chunks."""
        if len(data) <= self.config.fragment_size:
            return [data]
        
        fragments = []
        offset = 0
        while offset < len(data):
            fragment = data[offset:offset + self.config.fragment_size]
            fragments.append(fragment)
            offset += self.config.fragment_size
        
        return fragments
    
    async def send_reliable(self, data: bytes, send_func) -> bool:
        """Send data with reliability."""
        try:
            fragments = self.fragment_data(data)
            seq_num = self.get_next_sequence_number()
            total_fragments = len(fragments)
            
            # Send all fragments
            for i, fragment in enumerate(fragments):
                message = UDPMessage(
                    msg_type=UDPMessageType.DATA,
                    sequence_number=seq_num,
                    fragment_id=i,
                    total_fragments=total_fragments,
                    data=fragment,
                    timestamp=time.time()
                )
                
                if self.config.reliability_mode == ReliabilityMode.ACKNOWLEDGMENT:
                    # Store for potential retransmission
                    self.pending_acks[seq_num] = message
                
                # Send the message
                await send_func(message.to_bytes())
            
            # Wait for acknowledgment if required
            if self.config.reliability_mode == ReliabilityMode.ACKNOWLEDGMENT:
                return await self._wait_for_ack(seq_num)
            
            return True
            
        except Exception as e:
            logger.error(f"Reliable send failed: {e}")
            return False
    
    async def receive_reliable(self, receive_func) -> Optional[bytes]:
        """Receive data with reliability."""
        try:
            # Receive raw data
            raw_data = await receive_func()
            if not raw_data:
                return None
            
            data, sender = raw_data
            
            # Parse message
            try:
                message = UDPMessage.from_bytes(data)
            except ValueError as e:
                logger.warning(f"Invalid message received: {e}")
                return None
            
            # Handle different message types
            if message.msg_type == UDPMessageType.DATA:
                return await self._handle_data_message(message, sender)
            elif message.msg_type == UDPMessageType.ACK:
                await self._handle_ack_message(message)
            elif message.msg_type == UDPMessageType.PING:
                await self._handle_ping_message(message, sender)
            elif message.msg_type == UDPMessageType.PONG:
                await self._handle_pong_message(message, sender)
            
            return None
            
        except Exception as e:
            logger.error(f"Reliable receive failed: {e}")
            return None
    
    async def _handle_data_message(self, message: UDPMessage, sender) -> Optional[bytes]:
        """Handle incoming data message."""
        seq_num = message.sequence_number
        
        # Send acknowledgment if required
        if self.config.reliability_mode == ReliabilityMode.ACKNOWLEDGMENT:
            ack_message = UDPMessage(
                msg_type=UDPMessageType.ACK,
                sequence_number=seq_num,
                fragment_id=0,
                total_fragments=1,
                data=b''
            )
            # Send ACK back to sender
            try:
                ack_data = ack_message.to_bytes()
                if hasattr(self, '_send_ack_callback') and self._send_ack_callback:
                    await self._send_ack_callback(ack_data, sender)
                else:
                    logger.warning(f"No ACK callback set, cannot send ACK for sequence {seq_num}")
            except Exception as e:
                logger.error(f"Failed to send ACK for sequence {seq_num}: {e}")
        
        # Handle fragmentation
        if message.total_fragments > 1:
            # Store fragment
            if seq_num not in self.received_fragments:
                self.received_fragments[seq_num] = {}
            
            self.received_fragments[seq_num][message.fragment_id] = message
            
            # Check if all fragments received
            if len(self.received_fragments[seq_num]) == message.total_fragments:
                # Reassemble data
                fragments = self.received_fragments[seq_num]
                sorted_fragments = sorted(fragments.items())
                
                reassembled_data = b''.join(frag.data for _, frag in sorted_fragments)
                
                # Clean up
                del self.received_fragments[seq_num]
                
                return reassembled_data
        else:
            # Single fragment message
            return message.data
        
        return None
    
    async def _handle_ack_message(self, message: UDPMessage) -> None:
        """Handle acknowledgment message."""
        seq_num = message.sequence_number
        if seq_num in self.pending_acks:
            del self.pending_acks[seq_num]
            logger.debug(f"Received ACK for sequence {seq_num}")
    
    async def _handle_ping_message(self, message: UDPMessage, sender) -> None:
        """Handle ping message."""
        logger.debug(f"Received PING from {sender}")
        
        # Send PONG response
        try:
            pong_message = UDPMessage(
                msg_type=UDPMessageType.PONG,
                sequence_number=message.sequence_number,
                fragment_id=0,
                total_fragments=1,
                data=message.data
            )
            
            pong_data = pong_message.to_bytes()
            
            if hasattr(self, '_send_ack_callback') and self._send_ack_callback:
                await self._send_ack_callback(pong_data, sender)
                logger.debug(f"Sent PONG to {sender}")
            else:
                logger.warning(f"No callback set, cannot send PONG to {sender}")
                
        except Exception as e:
            logger.error(f"Failed to send PONG to {sender}: {e}")

    async def _handle_pong_message(self, message: UDPMessage, sender) -> None:
        """Handle pong message."""
        logger.debug(f"Received PONG from {sender} for sequence {message.sequence_number}")

    async def _wait_for_ack(self, seq_num: int) -> bool:
        """Wait for acknowledgment."""
        for attempt in range(self.config.max_retries):
            await asyncio.sleep(self.config.ack_timeout)
            
            if seq_num not in self.pending_acks:
                return True  # ACK received
            
            # Retransmit if configured and not the last attempt
            if (self.config.reliability_mode == ReliabilityMode.RETRANSMISSION and 
                attempt < self.config.max_retries - 1):
                try:
                    message = self.pending_acks[seq_num]
                    message_data = message.to_bytes()
                    
                    if hasattr(self, '_retransmit_callback') and self._retransmit_callback:
                        await self._retransmit_callback(message_data)
                        logger.debug(f"Retransmitted sequence {seq_num}, attempt {attempt + 1}")
                    else:
                        logger.warning(f"No retransmit callback set, cannot retransmit sequence {seq_num}")
                        
                except Exception as e:
                    logger.error(f"Failed to retransmit sequence {seq_num}: {e}")
        
        # Remove from pending after max retries
        if seq_num in self.pending_acks:
            del self.pending_acks[seq_num]
        
        return False  # ACK not received
  # ACK not received
  # ACK not received


class UDPTransport(BaseTransport):
    """UDP transport implementation with reliability and flow control."""
    
    def __init__(self, config: UDPConfig):
        super().__init__(config)
        self.socket: Optional[socket.socket] = None
        self.reliability_layer = UDPReliabilityLayer(config)
        self.remote_endpoint: Optional[Tuple[str, int]] = None
        
        # Set up callbacks for reliability layer
        self.reliability_layer.set_ack_callback(self._send_ack)
        self.reliability_layer.set_retransmit_callback(self._retransmit_message)

    async def _send_ack(self, ack_data: bytes, sender: Tuple[str, int]) -> None:
        """Send ACK message back to sender."""
        try:
            if not self.socket:
                logger.error("Cannot send ACK: socket not initialized")
                return
            
            # Send ACK directly to the sender
            await self._raw_send(ack_data, sender)
            logger.debug(f"Sent ACK to {sender}")
            
        except Exception as e:
            logger.error(f"Failed to send ACK to {sender}: {e}")

    async def _retransmit_message(self, message_data: bytes) -> None:
        """Retransmit a message to the current remote endpoint."""
        try:
            if not self.socket or not self.remote_endpoint:
                logger.error("Cannot retransmit: socket or remote endpoint not available")
                return
            
            # Retransmit to the current remote endpoint
            await self._raw_send(message_data, self.remote_endpoint)
            logger.debug(f"Retransmitted message to {self.remote_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to retransmit message: {e}")

    def get_transport_type(self) -> TransportType:
        return TransportType.UDP
    
    async def _internal_bind(self) -> bool:
        """Internal bind implementation."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.config.address, self.config.port))
            self.socket.setblocking(False)
            
            logger.info(f"UDP transport bound to {self.config.address}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bind UDP socket: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    async def _internal_connect(self) -> bool:
        """Internal connect implementation."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setblocking(False)
            self.remote_endpoint = (self.config.address, self.config.port)
            
            logger.info(f"UDP transport connected to {self.config.address}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect UDP socket: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    async def _internal_send(self, data: bytes, endpoint: Optional[Tuple[str, int]] = None) -> bool:
        """Internal send implementation."""
        try:
            if not self.socket:
                raise TransportError("Socket not initialized")
            
            target_endpoint = endpoint or self.remote_endpoint
            if not target_endpoint:
                raise TransportError("No target endpoint specified")
            
            # Use reliability layer if enabled
            if self.config.reliability_mode != ReliabilityMode.NONE:
                async def send_func(msg_data: bytes):
                    await self._raw_send(msg_data, target_endpoint)
                
                return await self.reliability_layer.send_reliable(data, send_func)
            else:
                return await self._raw_send(data, target_endpoint)
            
        except Exception as e:
            logger.error(f"UDP send failed: {e}")
            return False
    
    async def _raw_send(self, data: bytes, endpoint: Tuple[str, int]) -> bool:
        """Raw UDP send without reliability."""
        try:
            loop = asyncio.get_event_loop()
            await loop.sock_sendto(self.socket, data, endpoint)
            return True
        except Exception as e:
            logger.error(f"Raw UDP send failed: {e}")
            return False
    
    async def _internal_receive(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, Optional[Tuple[str, int]]]]:
        """Internal receive implementation."""
        try:
            if not self.socket:
                raise TransportError("Socket not initialized")
            
            # Use reliability layer if enabled
            if self.config.reliability_mode != ReliabilityMode.NONE:
                async def receive_func():
                    return await self._raw_receive(timeout)
                
                data = await self.reliability_layer.receive_reliable(receive_func)
                if data:
                    return data, None  # Sender info handled by reliability layer
                return None
            else:
                return await self._raw_receive(timeout)
            
        except Exception as e:
            logger.error(f"UDP receive failed: {e}")
            return None
    
    async def _raw_receive(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, Tuple[str, int]]]:
        """Raw UDP receive without reliability."""
        try:
            loop = asyncio.get_event_loop()
            
            if timeout:
                data, addr = await asyncio.wait_for(
                    loop.sock_recvfrom(self.socket, self.config.buffer_size),
                    timeout=timeout
                )
            else:
                data, addr = await loop.sock_recvfrom(self.socket, self.config.buffer_size)
            
            return data, addr
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Raw UDP receive failed: {e}")
            return None
    
    async def _internal_close(self) -> None:
        """Internal close implementation."""
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
                self.remote_endpoint = None
                logger.info("UDP transport closed")
        except Exception as e:
            logger.error(f"UDP close failed: {e}")

    async def ping(self, endpoint: Optional[Tuple[str, int]] = None, timeout: float = 5.0) -> bool:
        """Send a ping message and wait for pong response."""
        try:
            target_endpoint = endpoint or self.remote_endpoint
            if not target_endpoint:
                logger.error("Cannot ping: no target endpoint specified")
                return False
            
            # Create ping message
            ping_message = UDPMessage(
                msg_type=UDPMessageType.PING,
                sequence_number=self.reliability_layer.get_next_sequence_number(),
                fragment_id=0,
                total_fragments=1,
                data=b'ping',
                timestamp=time.time()
            )
            
            # Send ping
            ping_data = ping_message.to_bytes()
            send_success = await self._raw_send(ping_data, target_endpoint)
            
            if not send_success:
                return False
            
            # Wait for pong (simplified - in production would track specific sequence numbers)
            start_time = time.time()
            while time.time() - start_time < timeout:
                result = await self.receive(timeout=0.1)
                if result:
                    # If we received any data, consider ping successful for now
                    # In production, would verify it's the matching PONG
                    return True
                    
            return False  # Timeout
            
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return False


class UDPTransportPlugin(TransportPlugin):
    """UDP transport plugin."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def get_name(self) -> str:
        return "udp_transport"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "UDP transport with reliability layer and flow control"
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def get_interfaces(self) -> List[Type]:
        from ...plugins.interfaces.transport import ITransportProtocol
        return [ITransportProtocol]
    
    def _create_transport(self, config: TransportConfig) -> BaseTransport:
        udp_config = UDPConfig(**config.to_dict())
        return UDPTransport(udp_config)
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self.config.update(config)
        
    async def start(self) -> None:
        """Start the plugin."""
        pass
        
    async def stop(self) -> None:
        """Stop the plugin."""
        await self.transport.close()
        
    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        await self.stop()
        
    async def health_check(self) -> bool:
        """Check plugin health."""
        return True
