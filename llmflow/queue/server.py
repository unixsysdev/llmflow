"""
LLMFlow Queue Server

This module implements the queue server that handles client connections
and manages queues across the network.
"""

import asyncio
import signal
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import json
import argparse

from .protocol import QueueProtocol, QueueMessage, MessageType, SecurityLevel
from .manager import QueueManager, QueueConfig

logger = logging.getLogger(__name__)


class QueueServer:
    """LLMFlow Queue Server implementation."""
    
    def __init__(self, host: str = 'localhost', port: int = 8421, 
                 max_clients: int = 1000, log_level: str = 'INFO'):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.log_level = log_level
        
        # Core components
        self.protocol = QueueProtocol(host, port)
        self.manager = QueueManager(self.protocol)
        
        # Server state
        self.running = False
        self.start_time: Optional[datetime] = None
        self.client_connections: Set[str] = set()
        
        # Server configuration
        self.config = {
            'max_clients': max_clients,
            'default_queue_size': 10000,
            'default_message_ttl': 3600,  # 1 hour
            'cleanup_interval': 60,  # 1 minute
            'health_check_interval': 30,  # 30 seconds
            'metrics_interval': 60,  # 1 minute
        }
        
        # Metrics
        self.metrics = {
            'total_connections': 0,
            'active_connections': 0,
            'total_messages_processed': 0,
            'errors': 0,
            'uptime_seconds': 0,
            'queues_created': 0,
            'queues_deleted': 0
        }
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set specific loggers
        logging.getLogger('llmflow.queue').setLevel(self.log_level.upper())
    
    async def start(self) -> None:
        """Start the queue server."""
        if self.running:
            logger.warning("Server already running")
            return
        
        try:
            logger.info(f"Starting LLMFlow Queue Server on {self.host}:{self.port}")
            
            # Start queue manager
            await self.manager.start()
            
            # Create default queues
            await self._create_default_queues()
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._metrics_task = asyncio.create_task(self._metrics_loop())
            
            # Set running state
            self.running = True
            self.start_time = datetime.utcnow()
            
            logger.info(f"Queue server started successfully on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start queue server: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the queue server."""
        if not self.running:
            return
        
        logger.info("Stopping LLMFlow Queue Server")
        
        self.running = False
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        # Stop queue manager
        await self.manager.stop()
        
        # Clear client connections
        self.client_connections.clear()
        
        logger.info("Queue server stopped")
    
    async def _create_default_queues(self) -> None:
        """Create default system queues."""
        default_queues = [
            ('system.health', QueueConfig(
                queue_id='system.health',
                max_size=1000,
                persistent=True,
                domain='system'
            )),
            ('system.metrics', QueueConfig(
                queue_id='system.metrics',
                max_size=5000,
                persistent=True,
                domain='system'
            )),
            ('system.logs', QueueConfig(
                queue_id='system.logs',
                max_size=10000,
                persistent=True,
                domain='system'
            )),
            ('default.input', QueueConfig(
                queue_id='default.input',
                max_size=1000,
                domain='default'
            )),
            ('default.output', QueueConfig(
                queue_id='default.output',
                max_size=1000,
                domain='default'
            ))
        ]
        
        for queue_id, config in default_queues:
            success = await self.manager.create_queue(queue_id, config)
            if success:
                self.metrics['queues_created'] += 1
                logger.info(f"Created default queue: {queue_id}")
            else:
                logger.warning(f"Failed to create default queue: {queue_id}")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['health_check_interval'])
                
                # Perform health checks
                health_data = await self._perform_health_check()
                
                # Enqueue health data to system queue
                await self.manager.enqueue(
                    'system.health',
                    health_data,
                    domain='system'
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                self.metrics['errors'] += 1
    
    async def _metrics_loop(self) -> None:
        """Background metrics collection loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['metrics_interval'])
                
                # Collect metrics
                metrics_data = await self._collect_metrics()
                
                # Enqueue metrics to system queue
                await self.manager.enqueue(
                    'system.metrics',
                    metrics_data,
                    domain='system'
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                self.metrics['errors'] += 1
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        # Update uptime
        if self.start_time:
            self.metrics['uptime_seconds'] = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Get system stats
        system_stats = await self.manager.get_system_stats()
        
        # Get queue health
        queue_health = {}
        for queue_id in await self.manager.list_queues():
            stats = await self.manager.get_queue_stats(queue_id)
            if stats:
                queue_health[queue_id] = {
                    'message_count': stats.message_count,
                    'bytes_stored': stats.bytes_stored,
                    'last_activity': stats.last_activity.isoformat()
                }
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy' if self.running else 'unhealthy',
            'uptime_seconds': self.metrics['uptime_seconds'],
            'system_stats': system_stats,
            'queue_health': queue_health,
            'active_connections': len(self.client_connections)
        }
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'server_metrics': self.metrics.copy(),
            'system_stats': await self.manager.get_system_stats(),
            'protocol_metrics': self.protocol.get_metrics(),
            'config': self.config.copy()
        }
    
    async def create_queue(self, queue_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new queue via API."""
        if config:
            queue_config = QueueConfig(
                queue_id=queue_id,
                max_size=config.get('max_size', self.config['default_queue_size']),
                persistent=config.get('persistent', False),
                domain=config.get('domain', 'default'),
                tenant_id=config.get('tenant_id', 'default')
            )
        else:
            queue_config = QueueConfig(queue_id=queue_id)
        
        success = await self.manager.create_queue(queue_id, queue_config)
        if success:
            self.metrics['queues_created'] += 1
        
        return success
    
    async def delete_queue(self, queue_id: str) -> bool:
        """Delete a queue via API."""
        success = await self.manager.delete_queue(queue_id)
        if success:
            self.metrics['queues_deleted'] += 1
        
        return success
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            'server_id': f"{self.host}:{self.port}",
            'version': '0.1.0',
            'status': 'running' if self.running else 'stopped',
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': self.metrics['uptime_seconds'],
            'host': self.host,
            'port': self.port,
            'max_clients': self.max_clients,
            'active_clients': len(self.client_connections),
            'total_queues': len(await self.manager.list_queues()),
            'config': self.config.copy()
        }
    
    async def get_queue_list(self) -> List[Dict[str, Any]]:
        """Get list of all queues with statistics."""
        queue_list = []
        
        for queue_id in await self.manager.list_queues():
            stats = await self.manager.get_queue_stats(queue_id)
            if stats:
                queue_info = {
                    'queue_id': queue_id,
                    'message_count': stats.message_count,
                    'total_enqueued': stats.total_enqueued,
                    'total_dequeued': stats.total_dequeued,
                    'bytes_stored': stats.bytes_stored,
                    'created_at': stats.created_at.isoformat(),
                    'last_activity': stats.last_activity.isoformat()
                }
                queue_list.append(queue_info)
        
        return queue_list
    
    def add_client_connection(self, client_id: str) -> None:
        """Add a client connection."""
        self.client_connections.add(client_id)
        self.metrics['total_connections'] += 1
        self.metrics['active_connections'] = len(self.client_connections)
        logger.info(f"Client {client_id} connected")
    
    def remove_client_connection(self, client_id: str) -> None:
        """Remove a client connection."""
        self.client_connections.discard(client_id)
        self.metrics['active_connections'] = len(self.client_connections)
        logger.info(f"Client {client_id} disconnected")
    
    async def run_forever(self) -> None:
        """Run the server until interrupted."""
        await self.start()
        
        try:
            # Wait for shutdown signal
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()


def create_server_from_config(config_file: str) -> QueueServer:
    """Create a server from a configuration file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return QueueServer(
        host=config.get('host', 'localhost'),
        port=config.get('port', 8421),
        max_clients=config.get('max_clients', 1000),
        log_level=config.get('log_level', 'INFO')
    )


async def main():
    """Main server entry point."""
    parser = argparse.ArgumentParser(description='LLMFlow Queue Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8421, help='Server port')
    parser.add_argument('--max-clients', type=int, default=1000, help='Maximum clients')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create server
    if args.config:
        server = create_server_from_config(args.config)
    else:
        server = QueueServer(
            host=args.host,
            port=args.port,
            max_clients=args.max_clients,
            log_level=args.log_level
        )
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(server.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run server
    await server.run_forever()


if __name__ == '__main__':
    asyncio.run(main())
