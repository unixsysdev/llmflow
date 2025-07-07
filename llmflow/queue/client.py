"""
LLMFlow Queue Client

This module provides a client interface for interacting with LLMFlow queues.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import logging

from .protocol import QueueProtocol, QueueMessage, MessageType, SecurityLevel
from .manager import QueueManager, QueueConfig, QueueStats

logger = logging.getLogger(__name__)


class QueueClient:
    """Client for interacting with LLMFlow queues."""
    
    def __init__(self, host: str = 'localhost', port: int = 8421, 
                 default_domain: str = "default", default_tenant: str = "default"):
        self.host = host
        self.port = port
        self.default_domain = default_domain
        self.default_tenant = default_tenant
        self.protocol = QueueProtocol(host, port)
        self.connected = False
        self.client_id = str(uuid.uuid4())
        
        # Connection retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30.0
        
        # Client metrics
        self.metrics = {
            'operations_performed': 0,
            'errors': 0,
            'connected_at': None,
            'last_operation': None
        }
    
    async def connect(self) -> bool:
        """Connect to the queue system."""
        try:
            await self.protocol.start()
            self.connected = True
            self.metrics['connected_at'] = datetime.utcnow()
            logger.info(f"Queue client {self.client_id} connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect queue client: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the queue system."""
        if self.connected:
            await self.protocol.stop()
            self.connected = False
            logger.info(f"Queue client {self.client_id} disconnected")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def enqueue(self, queue_id: str, data: Dict[str, Any], 
                     domain: Optional[str] = None, tenant_id: Optional[str] = None,
                     security_level: SecurityLevel = SecurityLevel.PUBLIC) -> Optional[str]:
        """Enqueue data to a queue.
        
        Args:
            queue_id: ID of the target queue
            data: Data to enqueue
            domain: Domain context (defaults to client default)
            tenant_id: Tenant ID (defaults to client default)
            security_level: Security level for the message
            
        Returns:
            Message ID if successful, None otherwise
        """
        if not self.connected:
            logger.error("Client not connected")
            return None
        
        domain = domain or self.default_domain
        tenant_id = tenant_id or self.default_tenant
        
        try:
            message_id = await self.protocol.enqueue(
                queue_id=queue_id,
                data=data,
                domain=domain,
                tenant_id=tenant_id,
                security_level=security_level
            )
            
            self.metrics['operations_performed'] += 1
            self.metrics['last_operation'] = datetime.utcnow()
            
            logger.debug(f"Enqueued message {message_id} to queue {queue_id}")
            return message_id
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Failed to enqueue to {queue_id}: {e}")
            return None
    
    async def dequeue(self, queue_id: str, timeout: float = None) -> Optional[Dict[str, Any]]:
        """Dequeue data from a queue.
        
        Args:
            queue_id: ID of the source queue
            timeout: Maximum time to wait for a message
            
        Returns:
            Dequeued data if successful, None otherwise
        """
        if not self.connected:
            logger.error("Client not connected")
            return None
        
        timeout = timeout or self.timeout
        
        try:
            data = await self.protocol.dequeue(queue_id=queue_id, timeout=timeout)
            
            self.metrics['operations_performed'] += 1
            self.metrics['last_operation'] = datetime.utcnow()
            
            logger.debug(f"Dequeued message from queue {queue_id}")
            return data
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Failed to dequeue from {queue_id}: {e}")
            return None
    
    async def peek(self, queue_id: str) -> Optional[Dict[str, Any]]:
        """Peek at the next message in a queue without removing it.
        
        Args:
            queue_id: ID of the queue to peek at
            
        Returns:
            Next message data if available, None otherwise
        """
        if not self.connected:
            logger.error("Client not connected")
            return None
        
        try:
            data = await self.protocol.peek(queue_id=queue_id)
            
            self.metrics['operations_performed'] += 1
            self.metrics['last_operation'] = datetime.utcnow()
            
            logger.debug(f"Peeked at queue {queue_id}")
            return data
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Failed to peek at {queue_id}: {e}")
            return None
    
    async def transfer(self, from_queue: str, to_queue: str, count: int = 1) -> int:
        """Transfer messages from one queue to another.
        
        Args:
            from_queue: Source queue ID
            to_queue: Destination queue ID
            count: Number of messages to transfer
            
        Returns:
            Number of messages actually transferred
        """
        if not self.connected:
            logger.error("Client not connected")
            return 0
        
        try:
            transferred = await self.protocol.transfer(
                from_queue=from_queue,
                to_queue=to_queue,
                count=count
            )
            
            self.metrics['operations_performed'] += 1
            self.metrics['last_operation'] = datetime.utcnow()
            
            logger.info(f"Transferred {transferred} messages from {from_queue} to {to_queue}")
            return transferred
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Failed to transfer from {from_queue} to {to_queue}: {e}")
            return 0
    
    async def health_check(self) -> Optional[Dict[str, Any]]:
        """Get health status of the queue system.
        
        Returns:
            Health status information
        """
        if not self.connected:
            logger.error("Client not connected")
            return None
        
        try:
            health = await self.protocol.health_check()
            
            self.metrics['operations_performed'] += 1
            self.metrics['last_operation'] = datetime.utcnow()
            
            return health
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Failed to get health status: {e}")
            return None
    
    async def batch_enqueue(self, queue_id: str, data_list: List[Dict[str, Any]],
                           domain: Optional[str] = None, tenant_id: Optional[str] = None,
                           security_level: SecurityLevel = SecurityLevel.PUBLIC) -> List[Optional[str]]:
        """Enqueue multiple messages to a queue.
        
        Args:
            queue_id: ID of the target queue
            data_list: List of data items to enqueue
            domain: Domain context
            tenant_id: Tenant ID
            security_level: Security level for all messages
            
        Returns:
            List of message IDs (None for failed enqueues)
        """
        results = []
        
        for data in data_list:
            message_id = await self.enqueue(
                queue_id=queue_id,
                data=data,
                domain=domain,
                tenant_id=tenant_id,
                security_level=security_level
            )
            results.append(message_id)
        
        return results
    
    async def batch_dequeue(self, queue_id: str, count: int,
                           timeout: float = None) -> List[Optional[Dict[str, Any]]]:
        """Dequeue multiple messages from a queue.
        
        Args:
            queue_id: ID of the source queue
            count: Number of messages to dequeue
            timeout: Maximum time to wait for each message
            
        Returns:
            List of dequeued data items
        """
        results = []
        
        for _ in range(count):
            data = await self.dequeue(queue_id=queue_id, timeout=timeout)
            results.append(data)
            
            # Stop if we get None (queue empty)
            if data is None:
                break
        
        return results
    
    async def wait_for_message(self, queue_id: str, timeout: float = None) -> Optional[Dict[str, Any]]:
        """Wait for a message to become available in a queue.
        
        Args:
            queue_id: ID of the queue to monitor
            timeout: Maximum time to wait
            
        Returns:
            First available message or None if timeout
        """
        timeout = timeout or self.timeout
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            message = await self.peek(queue_id)
            if message:
                return await self.dequeue(queue_id)
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        return None
    
    async def get_client_metrics(self) -> Dict[str, Any]:
        """Get client-side metrics.
        
        Returns:
            Client metrics and statistics
        """
        return {
            'client_id': self.client_id,
            'connected': self.connected,
            'host': self.host,
            'port': self.port,
            'metrics': self.metrics.copy(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def set_retry_config(self, max_retries: int, retry_delay: float) -> None:
        """Configure retry behavior.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def set_timeout(self, timeout: float) -> None:
        """Set default timeout for operations.
        
        Args:
            timeout: Default timeout in seconds
        """
        self.timeout = timeout


class QueuePool:
    """Pool of queue clients for load balancing and failover."""
    
    def __init__(self, servers: List[Tuple[str, int]], 
                 default_domain: str = "default", default_tenant: str = "default"):
        self.servers = servers
        self.default_domain = default_domain
        self.default_tenant = default_tenant
        self.clients: List[QueueClient] = []
        self.current_client_index = 0
        self.pool_id = str(uuid.uuid4())
        
    async def initialize(self) -> None:
        """Initialize all clients in the pool."""
        for host, port in self.servers:
            client = QueueClient(
                host=host,
                port=port,
                default_domain=self.default_domain,
                default_tenant=self.default_tenant
            )
            
            if await client.connect():
                self.clients.append(client)
                logger.info(f"Added client {client.client_id} to pool")
            else:
                logger.warning(f"Failed to connect client for {host}:{port}")
    
    async def close(self) -> None:
        """Close all clients in the pool."""
        for client in self.clients:
            await client.disconnect()
        self.clients.clear()
    
    def get_next_client(self) -> Optional[QueueClient]:
        """Get the next available client using round-robin."""
        if not self.clients:
            return None
        
        # Round-robin selection
        client = self.clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        
        return client if client.connected else None
    
    async def enqueue(self, queue_id: str, data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Enqueue using the pool."""
        client = self.get_next_client()
        if client:
            return await client.enqueue(queue_id, data, **kwargs)
        return None
    
    async def dequeue(self, queue_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Dequeue using the pool."""
        client = self.get_next_client()
        if client:
            return await client.dequeue(queue_id, **kwargs)
        return None
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics for the entire pool."""
        stats = {
            'pool_id': self.pool_id,
            'total_clients': len(self.clients),
            'connected_clients': sum(1 for c in self.clients if c.connected),
            'servers': self.servers,
            'clients': []
        }
        
        for client in self.clients:
            client_stats = await client.get_client_metrics()
            stats['clients'].append(client_stats)
        
        return stats
