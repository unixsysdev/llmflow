#!/usr/bin/env python3
"""
LLMFlow Visual Interface Server

This module provides a web-based visual interface for designing, monitoring,
and managing LLMFlow applications. It includes a Node-RED/n8n style flow editor
specifically designed for LLMFlow's queue-based architecture.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# LLMFlow imports (optional for standalone mode)
try:
    from llmflow.core.base import Component, ComponentType
    from llmflow.queue.manager import QueueManager
    from llmflow.conductor.manager import ConductorManager
    from llmflow.master.optimizer import LLMOptimizer
    LLMFLOW_AVAILABLE = True
except ImportError:
    # Define minimal stubs for standalone mode
    Component = None
    ComponentType = None
    QueueManager = None
    ConductorManager = None
    LLMOptimizer = None
    LLMFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

# Data Models for API
class FlowDefinition(BaseModel):
    """Flow definition model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    connections: List[Dict[str, Any]] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)

class ComponentDefinition(BaseModel):
    """Component definition model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # atom, molecule, cell
    category: str
    description: str = ""
    input_types: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    icon: str = "default"
    color: str = "#3498db"

class DeploymentRequest(BaseModel):
    """Deployment request model."""
    flow_id: str
    environment: str = "dev"
    settings: Dict[str, Any] = Field(default_factory=dict)

class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]

class VisualInterfaceServer:
    """Main visual interface server."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(title="LLMFlow Visual Interface", version="1.0.0")
        
        # Storage for flows and components
        self.flows: Dict[str, FlowDefinition] = {}
        self.components: Dict[str, ComponentDefinition] = {}
        self.active_connections: Dict[str, WebSocket] = {}
        
        # LLMFlow system integration (optional)
        self.queue_manager = None
        self.conductor_manager = None
        self.optimizer = None
        self.llmflow_available = LLMFLOW_AVAILABLE
        
        self._setup_routes()
        self._setup_static_files()
        self._load_default_components()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_main_page():
            """Serve main application page."""
            return await self._get_index_html()
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
        
        # Flow Management API
        @self.app.get("/api/flows")
        async def list_flows():
            """List all flows."""
            return list(self.flows.values())
        
        @self.app.post("/api/flows")
        async def create_flow(flow: FlowDefinition):
            """Create a new flow."""
            self.flows[flow.id] = flow
            await self._broadcast_message("flow.created", {"flow": flow.dict()})
            return flow
        
        @self.app.get("/api/flows/{flow_id}")
        async def get_flow(flow_id: str):
            """Get a specific flow."""
            if flow_id not in self.flows:
                raise HTTPException(status_code=404, detail="Flow not found")
            return self.flows[flow_id]
        
        @self.app.put("/api/flows/{flow_id}")
        async def update_flow(flow_id: str, flow_data: Dict[str, Any]):
            """Update a flow."""
            if flow_id not in self.flows:
                raise HTTPException(status_code=404, detail="Flow not found")
            
            flow = self.flows[flow_id]
            for key, value in flow_data.items():
                if hasattr(flow, key):
                    setattr(flow, key, value)
            
            flow.updated_at = datetime.utcnow()
            await self._broadcast_message("flow.updated", {"flow": flow.dict()})
            return flow
        
        @self.app.delete("/api/flows/{flow_id}")
        async def delete_flow(flow_id: str):
            """Delete a flow."""
            if flow_id not in self.flows:
                raise HTTPException(status_code=404, detail="Flow not found")
            
            del self.flows[flow_id]
            await self._broadcast_message("flow.deleted", {"flow_id": flow_id})
            return {"status": "deleted"}
        
        # Component Management API
        @self.app.get("/api/components")
        async def list_components():
            """List all components."""
            return list(self.components.values())
        
        @self.app.post("/api/components")
        async def create_component(component: ComponentDefinition):
            """Create a new component."""
            self.components[component.id] = component
            await self._broadcast_message("component.created", {"component": component.dict()})
            return component
        
        @self.app.get("/api/components/{component_id}")
        async def get_component(component_id: str):
            """Get a specific component."""
            if component_id not in self.components:
                raise HTTPException(status_code=404, detail="Component not found")
            return self.components[component_id]
        
        # Deployment API
        @self.app.post("/api/deploy")
        async def deploy_flow(deployment: DeploymentRequest):
            """Deploy a flow to runtime."""
            if deployment.flow_id not in self.flows:
                raise HTTPException(status_code=404, detail="Flow not found")
            
            flow = self.flows[deployment.flow_id]
            deployment_id = str(uuid.uuid4())
            
            # TODO: Implement actual deployment logic
            deployment_result = {
                "deployment_id": deployment_id,
                "flow_id": deployment.flow_id,
                "environment": deployment.environment,
                "status": "deploying",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._broadcast_message("deployment.started", deployment_result)
            
            # Simulate deployment process
            asyncio.create_task(self._simulate_deployment(deployment_id, flow))
            
            return deployment_result
        
        # Monitoring API
        @self.app.get("/api/metrics")
        async def get_system_metrics():
            """Get system metrics."""
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "flows": {
                    "total": len(self.flows),
                    "active": len([f for f in self.flows.values() if f.settings.get("status") == "running"])
                },
                "components": {
                    "total": len(self.components),
                    "by_type": self._get_component_counts_by_type()
                },
                "system": {
                    "connected_clients": len(self.active_connections),
                    "uptime": "N/A",  # TODO: Calculate actual uptime
                    "memory_usage": "N/A"  # TODO: Get actual memory usage
                }
            }
            
            # Add conductor metrics if available
            if self.conductor_manager and hasattr(self.conductor_manager, 'get_conductor_status'):
                try:
                    conductor_status = self.conductor_manager.get_conductor_status()
                    metrics["conductor"] = conductor_status
                except Exception as e:
                    logger.warning(f"Failed to get conductor status: {e}")
            
            # Add optimizer metrics if available
            if self.optimizer and hasattr(self.optimizer, 'get_enhanced_optimizer_status'):
                try:
                    optimizer_status = self.optimizer.get_enhanced_optimizer_status()
                    metrics["optimizer"] = optimizer_status
                except Exception as e:
                    logger.warning(f"Failed to get optimizer status: {e}")
            
            # Add LLMFlow availability info
            metrics["llmflow_integration"] = {
                "available": self.llmflow_available,
                "components": {
                    "queue_manager": self.queue_manager is not None,
                    "conductor_manager": self.conductor_manager is not None,
                    "optimizer": self.optimizer is not None
                }
            }
            
            return metrics
        
        # WebSocket endpoint
        @self.app.websocket("/api/realtime")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication."""
            await websocket.accept()
            session_id = str(uuid.uuid4())
            self.active_connections[session_id] = websocket
            
            try:
                await websocket.send_text(json.dumps({
                    "type": "connection.established",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_websocket_message(session_id, message)
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {session_id}")
            finally:
                if session_id in self.active_connections:
                    del self.active_connections[session_id]
    
    def _setup_static_files(self):
        """Setup static file serving."""
        # Create static directory if it doesn't exist
        static_dir = Path("llmflow/visual/static")
        static_dir.mkdir(parents=True, exist_ok=True)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    def _load_default_components(self):
        """Load default component library."""
        # Data Atoms
        self.components["email_atom"] = ComponentDefinition(
            id="email_atom",
            name="Email Atom",
            type="atom",
            category="data",
            description="Email address data type with validation",
            output_types=["EmailAtom"],
            icon="mail",
            color="#3498db"
        )
        
        self.components["timestamp_atom"] = ComponentDefinition(
            id="timestamp_atom",
            name="Timestamp Atom",
            type="atom",
            category="data",
            description="Timestamp data type",
            output_types=["TimestampAtom"],
            icon="clock",
            color="#3498db"
        )
        
        # Service Atoms
        self.components["validate_email"] = ComponentDefinition(
            id="validate_email",
            name="Validate Email",
            type="atom",
            category="validation",
            description="Validates email address format",
            input_types=["EmailAtom"],
            output_types=["ValidationResult"],
            icon="check-circle",
            color="#27ae60"
        )
        
        self.components["hash_password"] = ComponentDefinition(
            id="hash_password",
            name="Hash Password",
            type="atom",
            category="security",
            description="Hash password using secure algorithm",
            input_types=["PasswordAtom"],
            output_types=["HashedPassword"],
            icon="lock",
            color="#27ae60"
        )
        
        # Molecules
        self.components["auth_molecule"] = ComponentDefinition(
            id="auth_molecule",
            name="Authentication Molecule",
            type="molecule",
            category="authentication",
            description="Complete user authentication workflow",
            input_types=["UserCredentials"],
            output_types=["AuthToken"],
            icon="user-check",
            color="#e67e22"
        )
        
        self.components["payment_molecule"] = ComponentDefinition(
            id="payment_molecule",
            name="Payment Molecule",
            type="molecule",
            category="payment",
            description="Payment processing workflow",
            input_types=["PaymentRequest"],
            output_types=["PaymentResult"],
            icon="credit-card",
            color="#e67e22"
        )
        
        # Cells
        self.components["user_management_cell"] = ComponentDefinition(
            id="user_management_cell",
            name="User Management Cell",
            type="cell",
            category="application",
            description="Complete user management system",
            input_types=["UserOperation"],
            output_types=["UserResult"],
            icon="users",
            color="#9b59b6"
        )
        
        # Infrastructure
        self.components["conductor"] = ComponentDefinition(
            id="conductor",
            name="Conductor",
            type="conductor",
            category="infrastructure",
            description="Process monitoring and management",
            input_types=["MonitoringData"],
            output_types=["ManagementAction"],
            icon="activity",
            color="#e74c3c"
        )
    
    async def _get_index_html(self) -> str:
        """Generate main application HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMFlow Visual Interface</title>
    <link rel="stylesheet" href="/static/css/main.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div id="app">
        <header class="header">
            <div class="header-left">
                <h1><i class="fas fa-project-diagram"></i> LLMFlow</h1>
                <span class="subtitle">Visual Flow Designer</span>
            </div>
            <div class="header-center">
                <div class="toolbar">
                    <button id="new-flow" class="btn btn-primary">
                        <i class="fas fa-plus"></i> New Flow
                    </button>
                    <button id="save-flow" class="btn btn-secondary">
                        <i class="fas fa-save"></i> Save
                    </button>
                    <button id="deploy-flow" class="btn btn-success">
                        <i class="fas fa-rocket"></i> Deploy
                    </button>
                    <button id="toggle-monitoring" class="btn btn-info">
                        <i class="fas fa-chart-line"></i> Monitor
                    </button>
                </div>
            </div>
            <div class="header-right">
                <div class="status-indicator">
                    <span class="status-dot status-connected"></span>
                    <span>Connected</span>
                </div>
            </div>
        </header>
        
        <div class="main-content">
            <aside class="sidebar">
                <div class="sidebar-tabs">
                    <button class="tab-btn active" data-tab="components">
                        <i class="fas fa-cube"></i> Components
                    </button>
                    <button class="tab-btn" data-tab="flows">
                        <i class="fas fa-project-diagram"></i> Flows
                    </button>
                    <button class="tab-btn" data-tab="monitoring">
                        <i class="fas fa-chart-bar"></i> Monitor
                    </button>
                </div>
                
                <div class="tab-content">
                    <div id="components-tab" class="tab-pane active">
                        <div class="search-box">
                            <input type="text" id="component-search" placeholder="Search components...">
                            <i class="fas fa-search"></i>
                        </div>
                        <div id="component-palette" class="component-palette">
                            <!-- Components will be loaded here -->
                        </div>
                    </div>
                    
                    <div id="flows-tab" class="tab-pane">
                        <div class="flow-list">
                            <div class="search-box">
                                <input type="text" id="flow-search" placeholder="Search flows...">
                                <i class="fas fa-search"></i>
                            </div>
                            <div id="flow-list" class="flows">
                                <!-- Flows will be loaded here -->
                            </div>
                        </div>
                    </div>
                    
                    <div id="monitoring-tab" class="tab-pane">
                        <div class="monitoring-panel">
                            <h3>System Status</h3>
                            <div id="system-metrics" class="metrics-grid">
                                <!-- Metrics will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            </aside>
            
            <main class="canvas-container">
                <div id="flow-canvas" class="flow-canvas">
                    <svg id="canvas-svg" class="canvas-svg">
                        <defs>
                            <!-- Arrow markers for connections -->
                            <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                                   refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
                            </marker>
                        </defs>
                        <g id="connections-layer"></g>
                    </svg>
                    <div id="nodes-layer" class="nodes-layer">
                        <!-- Flow nodes will be rendered here -->
                    </div>
                </div>
                
                <div class="canvas-controls">
                    <button id="zoom-in" class="control-btn">
                        <i class="fas fa-plus"></i>
                    </button>
                    <button id="zoom-out" class="control-btn">
                        <i class="fas fa-minus"></i>
                    </button>
                    <button id="zoom-fit" class="control-btn">
                        <i class="fas fa-expand"></i>
                    </button>
                    <button id="center-view" class="control-btn">
                        <i class="fas fa-crosshairs"></i>
                    </button>
                </div>
                
                <div id="minimap" class="minimap">
                    <canvas id="minimap-canvas"></canvas>
                </div>
            </main>
        </div>
        
        <div id="properties-panel" class="properties-panel">
            <div class="properties-header">
                <h3>Properties</h3>
                <button id="close-properties" class="close-btn">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div id="properties-content" class="properties-content">
                <!-- Property editors will be rendered here -->
            </div>
        </div>
        
        <!-- Modals -->
        <div id="flow-modal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 id="modal-title">New Flow</h3>
                    <button class="close-btn" onclick="closeModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <form id="flow-form">
                        <div class="form-group">
                            <label for="flow-name">Flow Name:</label>
                            <input type="text" id="flow-name" required>
                        </div>
                        <div class="form-group">
                            <label for="flow-description">Description:</label>
                            <textarea id="flow-description" rows="3"></textarea>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                    <button type="button" class="btn btn-primary" id="save-flow-btn">Create Flow</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="/static/js/app.js"></script>
</body>
</html>
"""
    
    async def _broadcast_message(self, message_type: str, data: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        message = {
            "type": message_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        disconnected_sessions = []
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send message to {session_id}: {e}")
                disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            del self.active_connections[session_id]
    
    async def _handle_websocket_message(self, session_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        message_type = message.get("type")
        data = message.get("data", {})
        
        if message_type == "ping":
            # Respond to ping with pong
            await self.active_connections[session_id].send_text(json.dumps({
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            }))
        
        elif message_type == "subscribe_metrics":
            # Subscribe to metrics updates
            # TODO: Implement metrics subscription
            pass
        
        elif message_type == "flow_update":
            # Handle flow updates from client
            flow_id = data.get("flow_id")
            if flow_id in self.flows:
                # Update flow and broadcast to other clients
                await self._broadcast_message("flow.updated", data)
    
    async def _simulate_deployment(self, deployment_id: str, flow: FlowDefinition):
        """Simulate deployment process."""
        await asyncio.sleep(2)  # Simulate deployment time
        
        deployment_result = {
            "deployment_id": deployment_id,
            "flow_id": flow.id,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoints": [
                f"http://localhost:8000/api/flows/{flow.id}/run"
            ]
        }
        
        await self._broadcast_message("deployment.completed", deployment_result)
    
    def _get_component_counts_by_type(self) -> Dict[str, int]:
        """Get count of components by type."""
        counts = {}
        for component in self.components.values():
            comp_type = component.type
            counts[comp_type] = counts.get(comp_type, 0) + 1
        return counts
    
    async def start(self, host: str = "localhost", port: int = 8080):
        """Start the visual interface server."""
        logger.info(f"Starting LLMFlow Visual Interface on {host}:{port}")
        
        # Initialize LLMFlow system connections if configured
        await self._initialize_llmflow_connections()
        
        # Start server
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def _initialize_llmflow_connections(self):
        """Initialize connections to LLMFlow system."""
        if not self.llmflow_available:
            logger.info("LLMFlow system not available - running in standalone mode")
            return
            
        try:
            # TODO: Initialize actual LLMFlow connections based on config
            if self.config.get("llmflow", {}).get("enabled", False):
                logger.info("LLMFlow integration enabled but not yet implemented")
            else:
                logger.info("LLMFlow integration disabled")
        except Exception as e:
            logger.warning(f"Could not initialize LLMFlow connections: {e}")

def main():
    """Main entry point for visual interface server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLMFlow Visual Interface Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
    
    # Create and start server
    server = VisualInterfaceServer(config)
    
    try:
        asyncio.run(server.start(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")

if __name__ == "__main__":
    main()
