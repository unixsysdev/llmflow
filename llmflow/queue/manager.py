"""
LLMFlow Queue Manager

This module implements the core queue management system for LLMFlow.
It handles queue creation, message storage, and queue operations.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
import logging
from contextlib import asynccontextmanager

from .protocol import QueueMessage, MessageType, SecurityLevel, QueueProtocol

logger = logging.getLogger(__name__)


@dataclass
class QueueStats:
    """Statistics for a queue."""
    queue_id: str
    message_count: int = 0
    total_enqueued: int = 0
    total_dequeued: int = 0
    bytes_stored: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


@dataclass
class QueueConfig:
    """Configuration for a queue."""
    queue_id: str
    max_size: int = 10000
    max_message_size: int = 1024 * 1024  # 1MB
    ttl: Optional[timedelta] = None
    persistent: bool = False
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    domain: str = "default"
    tenant_id: str = "default"
    
    def is_valid_message_size(self, size: int) -> bool:
        """Check if message size is valid."""
        return size <= self.max_message_size
    
    def is_queue_full(self, current_size: int) -> bool:
        """Check if queue is full."""
        return current_size >= self.max_size


class Queue:
    """Individual queue implementation."""
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self.messages: deque = deque()
        self.stats = QueueStats(queue_id=config.queue_id)
        self._lock = asyncio.Lock()
        self._subscribers: Set[str] = set()
        self._message_index: Dict[str, QueueMessage] = {}
        
    async def enqueue(self, message: QueueMessage) -> bool:
        """Add a message to the queue."""
        async with self._lock:
            # Check queue capacity
            if self.config.is_queue_full(len(self.messages)):
                logger.warning(f"Queue {self.config.queue_id} is full")
                return False
            
            # Check message size
            if not self.config.is_valid_message_size(len(message.payload)):
                logger.warning(f"Message too large for queue {self.config.queue_id}")
                return False
            
            # Check security level
            if message.security_level.value < self.config.security_level.value:
                logger.warning(f"Insufficient security level for queue {self.config.queue_id}")
                return False
            
            # Add message
            self.messages.append(message)
            self._message_index[message.message_id] = message
            
            # Update stats
            self.stats.message_count += 1
            self.stats.total_enqueued += 1
            self.stats.bytes_stored += len(message.payload)
            self.stats.update_activity()
            
            logger.debug(f"Enqueued message {message.message_id} to queue {self.config.queue_id}")
            return True
    
    async def dequeue(self) -> Optional[QueueMessage]:
        """Remove and return the next message from the queue."""
        async with self._lock:
            if not self.messages:
                return None
            
            message = self.messages.popleft()
            del self._message_index[message.message_id]
            
            # Update stats
            self.stats.message_count -= 1
            self.stats.total_dequeued += 1
            self.stats.bytes_stored -= len(message.payload)
            self.stats.update_activity()
            
            logger.debug(f"Dequeued message {message.message_id} from queue {self.config.queue_id}")
            return message
    
    async def peek(self) -> Optional[QueueMessage]:
        """Return the next message without removing it."""
        async with self._lock:
            if not self.messages:
                return None
            
            message = self.messages[0]
            self.stats.update_activity()
            
            logger.debug(f"Peeked message {message.message_id} from queue {self.config.queue_id}")
            return message
    
    async def get_message_by_id(self, message_id: str) -> Optional[QueueMessage]:
        """Get a specific message by ID."""
        async with self._lock:
            return self._message_index.get(message_id)
    
    async def size(self) -> int:
        """Get the current size of the queue."""
        return len(self.messages)
    
    async def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self.messages) == 0
    
    async def clear(self) -> int:
        """Clear all messages from the queue."""
        async with self._lock:
            count = len(self.messages)
            self.messages.clear()
            self._message_index.clear()
            
            # Reset stats
            self.stats.message_count = 0
            self.stats.bytes_stored = 0
            self.stats.update_activity()
            
            logger.info(f"Cleared {count} messages from queue {self.config.queue_id}")
            return count
    
    async def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        return self.stats
    
    def add_subscriber(self, subscriber_id: str) -> None:
        """Add a subscriber to the queue."""
        self._subscribers.add(subscriber_id)
    
    def remove_subscriber(self, subscriber_id: str) -> None:
        """Remove a subscriber from the queue."""
        self._subscribers.discard(subscriber_id)
    
    def get_subscribers(self) -> Set[str]:
        """Get all subscribers."""
        return self._subscribers.copy()


class QueueManager:
    """Central queue management system."""
    
    def __init__(self, protocol: Optional[QueueProtocol] = None):
        self.protocol = protocol or QueueProtocol()
        self.queues: Dict[str, Queue] = {}
        self.queue_configs: Dict[str, QueueConfig] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Register protocol handlers
        self.protocol.register_handler(MessageType.ENQUEUE, self._handle_enqueue)
        self.protocol.register_handler(MessageType.DEQUEUE, self._handle_dequeue)
        self.protocol.register_handler(MessageType.PEEK, self._handle_peek)
        self.protocol.register_handler(MessageType.TRANSFER, self._handle_transfer)
        self.protocol.register_handler(MessageType.HEALTH_CHECK, self._handle_health_check)
    
    async def start(self) -> None:
        """Start the queue manager."""
        if self._running:
            return
        
        await self.protocol.start()
        self._running = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Queue manager started")
    
    async def stop(self) -> None:
        """Stop the queue manager."""
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
        
        await self.protocol.stop()
        
        logger.info("Queue manager stopped")
    
    async def create_queue(self, queue_id: str, config: Optional[QueueConfig] = None) -> bool:
        """Create a new queue."""
        if not config:
            config = QueueConfig(queue_id=queue_id)
        
        async with self._lock:
            if queue_id in self.queues:
                logger.warning(f"Queue {queue_id} already exists")
                return False
            
            queue = Queue(config)
            self.queues[queue_id] = queue
            self.queue_configs[queue_id] = config
            
            logger.info(f"Created queue {queue_id}")
            return True
    
    async def delete_queue(self, queue_id: str) -> bool:
        """Delete a queue."""
        async with self._lock:
            if queue_id not in self.queues:
                logger.warning(f"Queue {queue_id} does not exist")
                return False
            
            # Clear the queue first
            await self.queues[queue_id].clear()
            
            # Remove from manager
            del self.queues[queue_id]
            del self.queue_configs[queue_id]
            
            logger.info(f"Deleted queue {queue_id}")
            return True
    
    async def get_queue(self, queue_id: str) -> Optional[Queue]:
        """Get a queue by ID."""
        return self.queues.get(queue_id)
    
    async def list_queues(self) -> List[str]:
        """List all queue IDs."""
        return list(self.queues.keys())
    
    async def enqueue(self, queue_id: str, data: Dict[str, Any], 
                     domain: str = "default", tenant_id: str = "default",
                     security_level: SecurityLevel = SecurityLevel.PUBLIC) -> Optional[str]:
        """Enqueue data to a queue."""
        queue = await self.get_queue(queue_id)
        if not queue:
            logger.warning(f"Queue {queue_id} does not exist")
            return None
        
        # Create message
        message_id = str(uuid.uuid4())
        import msgpack
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
        
        success = await queue.enqueue(message)
        return message_id if success else None
    
    async def dequeue(self, queue_id: str) -> Optional[Dict[str, Any]]:
        """Dequeue data from a queue."""
        queue = await self.get_queue(queue_id)
        if not queue:
            logger.warning(f"Queue {queue_id} does not exist")
            return None
        
        message = await queue.dequeue()
        if message:
            import msgpack
            return msgpack.unpackb(message.payload, raw=False)
        return None
    
    async def peek(self, queue_id: str) -> Optional[Dict[str, Any]]:
        """Peek at the next message in a queue."""
        queue = await self.get_queue(queue_id)
        if not queue:
            logger.warning(f"Queue {queue_id} does not exist")
            return None
        
        message = await queue.peek()
        if message:
            import msgpack
            return msgpack.unpackb(message.payload, raw=False)
        return None
    
    async def transfer(self, from_queue: str, to_queue: str, count: int = 1) -> int:
        """Transfer messages from one queue to another."""
        source_queue = await self.get_queue(from_queue)
        dest_queue = await self.get_queue(to_queue)
        
        if not source_queue:
            logger.warning(f"Source queue {from_queue} does not exist")
            return 0
        
        if not dest_queue:
            logger.warning(f"Destination queue {to_queue} does not exist")
            return 0
        
        transferred = 0
        for _ in range(count):
            message = await source_queue.dequeue()
            if message:
                # Update queue ID for transfer
                message.queue_id = to_queue
                success = await dest_queue.enqueue(message)
                if success:
                    transferred += 1
                else:
                    # Put message back if transfer failed
                    await source_queue.enqueue(message)
                    break
            else:
                break
        
        logger.info(f"Transferred {transferred} messages from {from_queue} to {to_queue}")
        return transferred
    
    async def get_queue_stats(self, queue_id: str) -> Optional[QueueStats]:
        """Get statistics for a queue."""
        queue = await self.get_queue(queue_id)
        if queue:
            return await queue.get_stats()
        return None
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        total_queues = len(self.queues)
        total_messages = sum(len(q.messages) for q in self.queues.values())
        total_bytes = sum(q.stats.bytes_stored for q in self.queues.values())
        
        return {
            'total_queues': total_queues,
            'total_messages': total_messages,
            'total_bytes': total_bytes,
            'protocol_metrics': self.protocol.get_metrics(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @asynccontextmanager
    async def queue_context(self, queue_id: str):
        """Context manager for queue operations."""
        queue = await self.get_queue(queue_id)
        if not queue:
            raise ValueError(f"Queue {queue_id} does not exist")
        
        try:
            yield queue
        finally:
            # Any cleanup logic here
            pass
    
    # Protocol message handlers
    async def _handle_enqueue(self, message: QueueMessage, addr) -> None:
        """Handle enqueue message."""
        # Implementation would send response
        pass
    
    async def _handle_dequeue(self, message: QueueMessage, addr) -> None:
        """Handle dequeue message."""
        # Implementation would send response
        pass
    
    async def _handle_peek(self, message: QueueMessage, addr) -> None:
        """Handle peek message."""
        # Implementation would send response
        pass
    
    async def _handle_transfer(self, message: QueueMessage, addr) -> None:
        """Handle transfer message."""
        # Implementation would send response
        pass
    
    async def _handle_health_check(self, message: QueueMessage, addr) -> None:
        """Handle health check message."""
        # Implementation would send response
        pass
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for expired messages and queues."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Cleanup expired messages
                current_time = datetime.utcnow()
                
                for queue_id, queue in list(self.queues.items()):
                    config = self.queue_configs[queue_id]
                    
                    if config.ttl:
                        # Remove expired messages
                        async with queue._lock:
                            expired_messages = []
                            for msg in queue.messages:
                                if current_time - msg.timestamp > config.ttl:
                                    expired_messages.append(msg)
                            
                            for msg in expired_messages:
                                queue.messages.remove(msg)
                                del queue._message_index[msg.message_id]
                                queue.stats.message_count -= 1
                                queue.stats.bytes_stored -= len(msg.payload)
                            
                            if expired_messages:
                                logger.info(f"Cleaned up {len(expired_messages)} expired messages from queue {queue_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
