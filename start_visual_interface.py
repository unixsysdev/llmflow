#!/usr/bin/env python3
"""
LLMFlow Visual Interface Startup Script

This script provides an easy way to start the LLMFlow visual interface server
with optional integration to the main LLMFlow system.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from llmflow.visual.server import VisualInterfaceServer
except ImportError as e:
    print(f"❌ Failed to import LLMFlow visual interface: {e}")
    print("Make sure you're in the LLMFlow project directory")
    sys.exit(1)

def setup_logging(level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('llmflow_visual.log')
        ]
    )

def load_config(config_path):
    """Load configuration from file."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)
    
    # Default configuration
    return {
        "server": {
            "host": "localhost",
            "port": 8080
        },
        "llmflow": {
            "enabled": False,
            "queue_manager_config": {},
            "conductor_config": {},
            "optimizer_config": {}
        },
        "features": {
            "realtime_collaboration": True,
            "auto_save": True,
            "metrics_collection": True
        }
    }

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLMFlow Visual Interface Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start with default settings
  %(prog)s --host 0.0.0.0 --port 8080  # Listen on all interfaces
  %(prog)s --config config.json    # Use custom configuration
  %(prog)s --log-level DEBUG       # Enable debug logging
        """
    )
    
    parser.add_argument(
        "--host", 
        default="localhost",
        help="Host to bind the server to (default: localhost)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080,
        help="Port to bind the server to (default: 8080)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Enable development mode with auto-reload"
    )
    
    parser.add_argument(
        "--llmflow-integration",
        action="store_true", 
        help="Enable LLMFlow system integration"
    )
    
    parser.add_argument(
        "--demo-data",
        action="store_true",
        help="Load demo flows and components for testing"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.host:
        config["server"]["host"] = args.host
    if args.port:
        config["server"]["port"] = args.port
    if args.llmflow_integration:
        config["llmflow"]["enabled"] = True
    if args.dev_mode:
        config["dev_mode"] = True
    if args.demo_data:
        config["demo_data"] = True
    
    # Create and configure server
    server = VisualInterfaceServer(config)
    
    # Load demo data if requested
    if args.demo_data:
        await load_demo_data(server)
    
    # Print startup information
    print("🎨 LLMFlow Visual Interface")
    print("=" * 40)
    print(f"Server: http://{config['server']['host']}:{config['server']['port']}")
    print(f"Log Level: {args.log_level}")
    print(f"LLMFlow Integration: {'Enabled' if config['llmflow']['enabled'] else 'Disabled'}")
    print(f"Development Mode: {'Enabled' if args.dev_mode else 'Disabled'}")
    print("=" * 40)
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        # Start the server
        await server.start(
            host=config["server"]["host"],
            port=config["server"]["port"]
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

async def load_demo_data(server):
    """Load demo flows and components for testing."""
    logger = logging.getLogger(__name__)
    logger.info("Loading demo data...")
    
    # Demo flows
    demo_flows = [
        {
            "name": "User Registration Flow",
            "description": "Complete user registration with email validation",
            "nodes": [
                {
                    "id": "email-input",
                    "componentId": "email_atom",
                    "x": 100,
                    "y": 100,
                    "properties": {}
                },
                {
                    "id": "email-validator",
                    "componentId": "validate_email",
                    "x": 300,
                    "y": 100,
                    "properties": {}
                },
                {
                    "id": "password-hash",
                    "componentId": "hash_password",
                    "x": 500,
                    "y": 100,
                    "properties": {}
                }
            ],
            "connections": [
                {
                    "id": "conn1",
                    "from": {"node": "email-input", "port": "output-0", "type": "EmailAtom"},
                    "to": {"node": "email-validator", "port": "input-0", "type": "EmailAtom"}
                }
            ],
            "settings": {"status": "stopped"}
        },
        {
            "name": "Payment Processing Flow", 
            "description": "Secure payment processing with validation",
            "nodes": [
                {
                    "id": "payment-request",
                    "componentId": "payment_molecule",
                    "x": 200,
                    "y": 200,
                    "properties": {}
                }
            ],
            "connections": [],
            "settings": {"status": "running"}
        }
    ]
    
    # Add demo flows to server
    for flow_data in demo_flows:
        from llmflow.visual.server import FlowDefinition
        flow = FlowDefinition(**flow_data)
        server.flows[flow.id] = flow
    
    logger.info(f"Loaded {len(demo_flows)} demo flows")

def create_sample_config():
    """Create a sample configuration file."""
    config = {
        "server": {
            "host": "localhost",
            "port": 8080
        },
        "llmflow": {
            "enabled": True,
            "queue_manager_config": {
                "transport": "udp",
                "host": "localhost",
                "port": 8421
            },
            "conductor_config": {
                "health_check_interval": 30.0,
                "metrics_collection_interval": 60.0
            },
            "optimizer_config": {
                "enable_predictive_optimization": True,
                "enable_multi_component_analysis": True
            }
        },
        "features": {
            "realtime_collaboration": True,
            "auto_save": True,
            "metrics_collection": True,
            "demo_mode": False
        },
        "ui": {
            "theme": "dark",
            "auto_layout": True,
            "grid_snap": True,
            "minimap": True
        }
    }
    
    with open("visual_interface_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("📄 Sample configuration created: visual_interface_config.json")

if __name__ == "__main__":
    # Special case for creating sample config
    if len(sys.argv) == 2 and sys.argv[1] == "--create-config":
        create_sample_config()
        sys.exit(0)
    
    # Run the main application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
