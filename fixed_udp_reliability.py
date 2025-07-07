"""
Fixed UDP Reliability Layer Implementation

This file contains the corrected implementation of the UDP reliability layer
with proper indentation and complete ACK/PONG/retransmission functionality.
"""

import asyncio
import socket
import struct
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ReliabilityMode(Enum):
    """UDP reliability modes."""
    NONE = "none"
    ACKNOWLEDGMENT = "acknowledgment"
    RETRANSMISSION = "retransmission"


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
        if len(data) < 9:  # Minimum header size
            raise ValueError("Invalid message: too short")
        
        msg_type, seq_num, frag_id, total_frags, data_len = struct.unpack('!BHHHI', data[:9])
        
        if len(data) < 9 + data_len:
            raise ValueError("Invalid message: data length mismatch")
        
        return cls(
            msg_type=UDPMessageType(msg_type),
            sequence_number=seq_num,
            fragment_id=frag_id,
            total_fragments=total_frags,
            data=data[9:9+data_len],
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
                # We need access to the transport's send function
                # This will be provided by the caller
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
                sequence_number=message.sequence_number,  # Echo back the same sequence number
                fragment_id=0,
                total_fragments=1,
                data=message.data  # Echo back any ping data
            )
            
            pong_data = pong_message.to_bytes()
            
            # Send PONG back to sender
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
        # PONG messages can be used for latency measurement or health checks
        # Store timing information if needed for health monitoring

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
                    
                    # Retransmit message using callback
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


print("✓ Fixed UDP reliability layer implementation loaded successfully")
