"""
LLMFlow Queue Communication Protocol

This module implements the core queue communication protocol for LLMFlow.
The protocol is UDP-based with reliability features inspired by HTTP3/QUIC.
"""

import asyncio
import struct
import uuid
from enum import IntEnum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import msgpack
import hashlib
import logging

logger = logging.getLogger(__name__)


class MessageType(IntEnum):
    """Message types for queue protocol."""
    ENQUEUE = 1
    DEQUEUE = 2
    PEEK = 3
    TRANSFER = 4
    CONTEXT_SWITCH = 5
    HEALTH_CHECK = 6
    METRICS = 7
    ACK = 8
    NACK = 9
    HEARTBEAT = 10


class SecurityLevel(IntEnum):
    """Security levels for queue contexts."""
    PUBLIC = 0
    AUTHENTICATED = 1
    ENCRYPTED = 2
    CONFIDENTIAL = 3


@dataclass
class QueueMessage:
    """Represents a message in the queue system."""
    message_id: str
    queue_id: str
    message_type: MessageType
    payload: bytes
    security_level: SecurityLevel
    domain: str
    tenant_id: str
    timestamp: datetime
    checksum: str
    signature: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of message payload."""
        return hashlib.sha256(self.payload).hexdigest()
    
    def is_valid(self) -> bool:
        """Verify message integrity."""
        return self.checksum == self._calculate_checksum()
    
    def serialize(self) -> bytes:
        """Serialize message to bytes using the protocol format."""
        # Header: MessageType(1) + QueueID(8) + MessageID(8) + PayloadSize(4)
        queue_id_bytes = self.queue_id.encode('utf-8')[:8].ljust(8, b'\x00')
        message_id_bytes = self.message_id.encode('utf-8')[:8].ljust(8, b'\x00')
        payload_size = len(self.payload)
        
        header = struct.pack(
            '!B8s8sI',
            self.message_type.value,
            queue_id_bytes,
            message_id_bytes,
            payload_size
        )
        
        # Context: SecurityLevel(1) + Domain(variable) + TenantID(variable)
        domain_bytes = self.domain.encode('utf-8')
        tenant_id_bytes = self.tenant_id.encode('utf-8')
        
        context = struct.pack(
            f'!BH{len(domain_bytes)}sH{len(tenant_id_bytes)}s',
            self.security_level.value,
            len(domain_bytes),
            domain_bytes,
            len(tenant_id_bytes),
            tenant_id_bytes
        )
        
        return header + context + self.payload
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'QueueMessage':
        """Deserialize bytes to QueueMessage."""
        # Parse header
        message_type, queue_id_bytes, message_id_bytes, payload_size = struct.unpack(
            '!B8s8sI', data[:21]
        )
        
        queue_id = queue_id_bytes.decode('utf-8').rstrip('\x00')
        message_id = message_id_bytes.decode('utf-8').rstrip('\x00')
        
        # Parse context
        offset = 21
        security_level = struct.unpack('!B', data[offset:offset+1])[0]
        offset += 1
        
        domain_len = struct.unpack('!H', data[offset:offset+2])[0]
        offset += 2
        domain = data[offset:offset+domain_len].decode('utf-8')
        offset += domain_len
        
        tenant_id_len = struct.unpack('!H', data[offset:offset+2])[0]
        offset += 2
        tenant_id = data[offset:offset+tenant_id_len].decode('utf-8')
        offset += tenant_id_len
        
        # Extract payload
        payload = data[offset:offset+payload_size]
        
        return cls(
            message_id=message_id,
            queue_id=queue_id,
            message_type=MessageType(message_type),
            payload=payload,
            security_level=SecurityLevel(security_level),
            domain=domain,
            tenant_id=tenant_id,
            timestamp=datetime.utcnow(),
            checksum=""  # Will be calculated in __post_init__
        )


@dataclass
class QueueOperation:
    """Represents a queue operation request."""
    operation_id: str
    message_type: MessageType
    queue_id: str
    data: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        """Set operation ID if not provided."""
        if not self.operation_id:
            self.operation_id = str(uuid.uuid4())


class QueueProtocol:
    """Core queue protocol implementation."""
    
    def __init__(self, host: str = 'localhost', port: int = 8421):
        self.host = host
        self.port = port
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[asyncio.DatagramProtocol] = None
        self.pending_operations: Dict[str, asyncio.Future] = {}
        self.message_handlers: Dict[MessageType, callable] = {}
        self.running = False
        
        # Performance metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'average_latency': 0.0
        }
    
    async def start(self) -> None:
        """Start the queue protocol."""
        loop = asyncio.get_event_loop()
        
        # Create UDP transport
        self.transport, self.protocol = await loop.create_datagram_endpoint(
            lambda: QueueProtocolHandler(self),
            local_addr=(self.host, self.port)
        )
        
        self.running = True
        logger.info(f"Queue protocol started on {self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Stop the queue protocol."""
        self.running = False
        
        # Cancel pending operations
        for future in self.pending_operations.values():
            future.cancel()
        
        if self.transport:
            self.transport.close()
        
        logger.info("Queue protocol stopped")
    
    async def send_message(self, message: QueueMessage, 
                          destination: Tuple[str, int]) -> bool:
        """Send a message to a destination."""
        if not self.transport:
            raise RuntimeError("Protocol not started")
        
        try:
            data = message.serialize()
            self.transport.sendto(data, destination)
            
            self.metrics['messages_sent'] += 1
            logger.debug(f"Sent message {message.message_id} to {destination}")
            return True
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def enqueue(self, queue_id: str, data: Dict[str, Any], 
                     domain: str = "default", tenant_id: str = "default",
                     security_level: SecurityLevel = SecurityLevel.PUBLIC) -> str:
        """Enqueue data to a queue."""
        message_id = str(uuid.uuid4())
        payload = msgpack.packb(data)
        
        message = QueueMessage(
            message_id=message_id,
            queue_id=queue_id,
            message_type=MessageType.ENQUEUE,
            payload=payload,
            security_level=security_level,
            domain=domain,
            tenant_id=tenant_id,
            timestamp=datetime.utcnow(),
            checksum=""
        )
        
        # For now, assume local processing
        # In a real implementation, this would send to a queue server
        await self._process_enqueue(message)
        
        return message_id
    
    async def dequeue(self, queue_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Dequeue data from a queue."""
        message_id = str(uuid.uuid4())
        
        message = QueueMessage(
            message_id=message_id,
            queue_id=queue_id,
            message_type=MessageType.DEQUEUE,
            payload=b'',
            security_level=SecurityLevel.PUBLIC,
            domain="default",
            tenant_id="default",
            timestamp=datetime.utcnow(),
            checksum=""
        )
        
        # For now, assume local processing
        result = await self._process_dequeue(message)
        
        if result:
            return msgpack.unpackb(result, raw=False)
        return None
    
    async def peek(self, queue_id: str) -> Optional[Dict[str, Any]]:
        """Peek at the next message in a queue without removing it."""
        message_id = str(uuid.uuid4())
        
        message = QueueMessage(
            message_id=message_id,
            queue_id=queue_id,
            message_type=MessageType.PEEK,
            payload=b'',
            security_level=SecurityLevel.PUBLIC,
            domain="default",
            tenant_id="default",
            timestamp=datetime.utcnow(),
            checksum=""
        )
        
        result = await self._process_peek(message)
        
        if result:
            return msgpack.unpackb(result, raw=False)
        return None
    
    async def transfer(self, from_queue: str, to_queue: str, 
                      count: int = 1) -> int:
        """Transfer messages from one queue to another."""
        transferred = 0
        
        for _ in range(count):
            message = await self.dequeue(from_queue)
            if message:
                await self.enqueue(to_queue, message)
                transferred += 1
            else:
                break
        
        return transferred
    
    async def health_check(self) -> Dict[str, Any]:
        """Get health status of the queue system."""
        return {
            'status': 'healthy' if self.running else 'stopped',
            'metrics': self.metrics.copy(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Internal processing methods (would be implemented by queue manager)
    async def _process_enqueue(self, message: QueueMessage) -> None:
        """Process an enqueue message."""
        # This would be implemented by the queue manager
        pass
    
    async def _process_dequeue(self, message: QueueMessage) -> Optional[bytes]:
        """Process a dequeue message."""
        # This would be implemented by the queue manager
        return None
    
    async def _process_peek(self, message: QueueMessage) -> Optional[bytes]:
        """Process a peek message."""
        # This would be implemented by the queue manager
        return None
    
    def register_handler(self, message_type: MessageType, handler: callable) -> None:
        """Register a message handler."""
        self.message_handlers[message_type] = handler
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get protocol metrics."""
        return self.metrics.copy()


class QueueProtocolHandler(asyncio.DatagramProtocol):
    """UDP protocol handler for queue messages."""
    
    def __init__(self, queue_protocol: QueueProtocol):
        self.queue_protocol = queue_protocol
    
    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        """Called when connection is established."""
        self.transport = transport
        logger.debug("Queue protocol connection established")
    
    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Called when a datagram is received."""
        try:
            message = QueueMessage.deserialize(data)
            
            if not message.is_valid():
                logger.warning(f"Invalid message received from {addr}")
                return
            
            self.queue_protocol.metrics['messages_received'] += 1
            
            # Handle message based on type
            handler = self.queue_protocol.message_handlers.get(message.message_type)
            if handler:
                asyncio.create_task(handler(message, addr))
            else:
                logger.warning(f"No handler for message type {message.message_type}")
                
        except Exception as e:
            logger.error(f"Error processing datagram from {addr}: {e}")
            self.queue_protocol.metrics['errors'] += 1
    
    def error_received(self, exc: Exception) -> None:
        """Called when an error occurs."""
        logger.error(f"Queue protocol error: {exc}")
        self.queue_protocol.metrics['errors'] += 1
    
    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Called when connection is lost."""
        logger.info("Queue protocol connection lost")
        if exc:
            logger.error(f"Connection lost due to error: {exc}")
