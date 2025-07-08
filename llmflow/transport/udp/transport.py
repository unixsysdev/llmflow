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
from .reliability import ReliabilityManager

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




# NOTE: UDPReliabilityLayer has been replaced by ReliabilityManager
# The old implementation is commented out as it's no longer used.
# See llmflow/transport/udp/reliability.py for the new implementation.

# class UDPReliabilityLayer:
#     [Old implementation removed - replaced by ReliabilityManager]




class UDPTransport(BaseTransport):
    """UDP transport implementation with reliability and flow control."""
    
    def __init__(self, config: UDPConfig):
        super().__init__(config)
        self.socket: Optional[socket.socket] = None
        # FIXED: Use ReliabilityManager instead of UDPReliabilityLayer for better reliability
        self.reliability_manager = ReliabilityManager(
            max_retries=config.max_retries,
            ack_timeout=config.timeout,
            enable_flow_control=True
        )
        self.remote_endpoint: Optional[Tuple[str, int]] = None


    # NOTE: _send_ack and _retransmit_message methods removed
# These were callbacks for the old UDPReliabilityLayer.
# ACK handling is now integrated directly in _internal_receive method.
# Retransmission is handled by ReliabilityManager.

    def get_transport_type(self) -> TransportType:
        return TransportType.UDP
    
    async def _internal_bind(self) -> bool:
        """Internal bind implementation."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.config.address, self.config.port))
            self.socket.setblocking(False)
            
            # Start reliability manager
            await self.reliability_manager.start()
            
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
            
            # Start reliability manager
            await self.reliability_manager.start()
            
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
            
            # Use reliability manager if enabled
            if self.config.reliability_mode != ReliabilityMode.NONE:
                sequence_number = self.reliability_manager.get_next_sequence_number()
                
                # Create UDP message
                udp_message = UDPMessage(
                    msg_type=UDPMessageType.DATA,
                    sequence_number=sequence_number,
                    fragment_id=0,
                    total_fragments=1,
                    data=data
                )
                
                async def send_func(msg_data: bytes):
                    # Send the formatted UDP message
                    await self._raw_send(udp_message.to_bytes(), target_endpoint)
                
                return await self.reliability_manager.send_reliable(data, send_func)
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
            
            # Use reliability manager if enabled
            if self.config.reliability_mode != ReliabilityMode.NONE:
                # Receive raw data and parse UDP messages
                raw_result = await self._raw_receive(timeout)
                if not raw_result:
                    return None
                
                raw_data, sender_addr = raw_result
                
                try:
                    # Parse UDP message
                    udp_message = UDPMessage.from_bytes(raw_data)
                    
                    # Handle ACK messages
                    if udp_message.msg_type == UDPMessageType.ACK:
                        await self.reliability_manager.handle_ack(udp_message.sequence_number)
                        return None  # ACKs are not returned to application
                    
                    # Handle data messages through reliability manager
                    elif udp_message.msg_type == UDPMessageType.DATA:
                        is_duplicate, is_out_of_order = await self.reliability_manager.handle_received_message(
                            udp_message.sequence_number, udp_message.data
                        )
                        
                        # Send ACK for this message
                        ack_message = UDPMessage(
                            msg_type=UDPMessageType.ACK,
                            sequence_number=udp_message.sequence_number,
                            fragment_id=0,
                            total_fragments=1,
                            data=b""
                        )
                        await self._raw_send(ack_message.to_bytes(), sender_addr)
                        
                        # Return data if not duplicate
                        if not is_duplicate:
                            return udp_message.data, sender_addr
                        else:
                            return None  # Skip duplicate messages
                    
                    else:
                        # Handle other message types (PING, PONG, etc.)
                        return udp_message.data, sender_addr
                        
                except ValueError as e:
                    logger.warning(f"Failed to parse UDP message: {e}")
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
            # Stop reliability manager
            await self.reliability_manager.stop()
            
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
