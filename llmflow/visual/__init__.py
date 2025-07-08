"""
LLMFlow Visual Interface Package

This package provides a comprehensive web-based visual interface for designing,
monitoring, and managing LLMFlow applications. It includes:

- Visual flow designer with drag-and-drop components
- Real-time monitoring dashboard
- Component library and palette
- Flow deployment and management
- WebSocket-based real-time updates
- Integration with LLMFlow core systems

Usage:
    from llmflow.visual.server import VisualInterfaceServer
    
    server = VisualInterfaceServer(config)
    await server.start()
"""

__version__ = "1.0.0"
__author__ = "LLMFlow Team"

from .server import VisualInterfaceServer

__all__ = [
    "VisualInterfaceServer"
]
