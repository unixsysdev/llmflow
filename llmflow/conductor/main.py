"""
LLMFlow Conductor Main

This module provides the main entry point for the LLMFlow conductor system.
"""

import asyncio
import signal
import logging
import argparse
import json
from typing import Dict, Any
from pathlib import Path

from .manager import ConductorManager
from .monitor import ConductorMonitor
from ..queue import QueueManager, QueueClient
from ..core.base import Component, ComponentType

logger = logging.getLogger(__name__)


class ConductorService:
    """Main conductor service that coordinates management and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.queue_manager = QueueManager()
        self.manager = ConductorManager(self.queue_manager)
        self.monitor = ConductorMonitor(self.queue_manager)
        
        # Service state
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        self._setup_logging()
        
        # Register signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config.get('log_level', 'INFO')
        log_format = self.config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )
        
        # Set specific loggers
        logging.getLogger('llmflow.conductor').setLevel(log_level.upper())
        logging.getLogger('llmflow.queue').setLevel(log_level.upper())
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self) -> None:
        """Start the conductor service."""
        if self.running:
            logger.warning("Conductor service already running")
            return
        
        try:
            logger.info("Starting LLMFlow Conductor Service")
            
            # Start queue manager
            await self.queue_manager.start()
            
            # Start conductor manager
            await self.manager.start()
            
            # Start monitor
            await self.monitor.start()
            
            # Register monitor callbacks
            self.monitor.add_alert_callback(self._handle_alert)
            
            # Register example components if configured
            if self.config.get('register_examples', False):
                await self._register_example_components()
            
            self.running = True
            logger.info("Conductor service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start conductor service: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the conductor service."""
        if not self.running:
            return
        
        logger.info("Stopping LLMFlow Conductor Service")
        
        self.running = False
        
        # Stop components in reverse order
        await self.monitor.stop()
        await self.manager.stop()
        await self.queue_manager.stop()
        
        logger.info("Conductor service stopped")
    
    async def run(self) -> None:
        """Run the conductor service until shutdown."""
        await self.start()
        
        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
        except Exception as e:
            logger.error(f"Error in conductor service: {e}")
        finally:
            await self.stop()
    
    async def _register_example_components(self) -> None:
        """Register example components for demonstration."""
        # Create example components
        example_components = [
            Component(
                name="auth_service",
                component_type=ComponentType.SERVICE_ATOM
            ),
            Component(
                name="validation_service",
                component_type=ComponentType.SERVICE_ATOM
            ),
            Component(
                name="data_processor",
                component_type=ComponentType.MOLECULE
            )
        ]
        
        # Register components with conductor
        for component in example_components:
            # Initialize component
            component.initialize({'example': True})
            
            # Register with manager
            process_id = await self.manager.register_process(
                component,
                queue_bindings=[f"{component.name}.input", f"{component.name}.output"]
            )
            
            # Start the process
            await self.manager.start_process(process_id)
            
            logger.info(f"Registered and started example component: {component.name}")
    
    async def _handle_alert(self, alert) -> None:
        """Handle triggered alerts."""
        logger.warning(f"ALERT: {alert.message} (severity: {alert.severity})")
        
        # In production, this might send notifications, trigger actions, etc.
        # For now, just log the alert
        
        # Example: Auto-restart processes on critical alerts
        if alert.severity == 'critical':
            logger.info("Critical alert detected, checking for processes to restart...")
            
            # Get all process info
            processes = await self.manager.list_processes()
            
            # Find failed processes
            failed_processes = [p for p in processes if p.status.value == 'failed']
            
            # Restart failed processes
            for process in failed_processes:
                logger.info(f"Restarting failed process: {process.process_id}")
                await self.manager.restart_process(process.process_id)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        manager_metrics = await self.manager.get_conductor_metrics()
        monitor_status = await self.monitor.get_monitor_status()
        
        return {
            'service': {
                'running': self.running,
                'config': self.config
            },
            'manager': manager_metrics,
            'monitor': monitor_status,
            'queue_manager': await self.queue_manager.get_system_stats()
        }
    
    async def get_processes(self) -> Dict[str, Any]:
        """Get information about managed processes."""
        processes = await self.manager.list_processes()
        
        return {
            'processes': [
                {
                    'process_id': p.process_id,
                    'component_name': p.component_name,
                    'component_type': p.component_type.value,
                    'status': p.status.value,
                    'pid': p.pid,
                    'started_at': p.started_at.isoformat() if p.started_at else None,
                    'restart_count': p.restart_count,
                    'queue_bindings': p.queue_bindings,
                    'is_healthy': p.is_healthy()
                }
                for p in processes
            ],
            'total_processes': len(processes),
            'running_processes': len([p for p in processes if p.status.value == 'running']),
            'failed_processes': len([p for p in processes if p.status.value == 'failed'])
        }
    
    async def get_alerts(self) -> Dict[str, Any]:
        """Get alert information."""
        active_alerts = await self.monitor.get_active_alerts()
        alert_history = await self.monitor.get_alert_history()
        
        return {
            'active_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'rule_id': alert.rule_id,
                    'severity': alert.severity,
                    'message': alert.message,
                    'triggered_at': alert.triggered_at.isoformat(),
                    'metadata': alert.metadata
                }
                for alert in active_alerts
            ],
            'alert_history': [
                {
                    'alert_id': alert.alert_id,
                    'rule_id': alert.rule_id,
                    'severity': alert.severity,
                    'message': alert.message,
                    'triggered_at': alert.triggered_at.isoformat(),
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                    'metadata': alert.metadata
                }
                for alert in alert_history
            ],
            'total_active': len(active_alerts),
            'total_history': len(alert_history)
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Merge with defaults
        default_config = get_default_config()
        default_config.update(config)
        
        return default_config
    
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'log_level': 'INFO',
        'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'register_examples': False,
        'queue_manager': {
            'host': 'localhost',
            'port': 8421
        },
        'conductor': {
            'health_check_interval': 30.0,
            'metrics_collection_interval': 60.0,
            'restart_check_interval': 10.0,
            'max_concurrent_restarts': 3
        },
        'monitor': {
            'aggregation_interval': 60.0,
            'alert_check_interval': 10.0,
            'cleanup_interval': 3600.0,
            'metrics_retention_hours': 24,
            'alert_retention_days': 7
        }
    }


async def main():
    """Main entry point for the conductor service."""
    parser = argparse.ArgumentParser(description='LLMFlow Conductor Service')
    parser.add_argument('--config', default='config/conductor.json', help='Configuration file path')
    parser.add_argument('--log-level', help='Override log level')
    parser.add_argument('--register-examples', action='store_true', help='Register example components')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.log_level:
        config['log_level'] = args.log_level
    
    if args.register_examples:
        config['register_examples'] = True
    
    # Create and run service
    service = ConductorService(config)
    
    try:
        await service.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Conductor service error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
