# LLMFlow Visual Interface

A comprehensive web-based visual interface for designing, monitoring, and managing LLMFlow applications. This interface provides a Node-RED/n8n style flow designer specifically tailored for LLMFlow's queue-based architecture.

## Features

### 🎨 Visual Flow Designer
- **Drag-and-Drop Interface**: Intuitive component placement from palette to canvas
- **Node-RED Style**: Familiar visual programming interface
- **Type-Safe Connections**: Visual validation of component type compatibility
- **Real-time Collaboration**: Multiple users can edit flows simultaneously
- **Canvas Controls**: Zoom, pan, fit-to-screen, and center view

### 📚 Component Library
- **Categorized Palette**: Components organized by type (atoms, molecules, cells)
- **Searchable Components**: Find components quickly with search functionality
- **Component Properties**: Dynamic property editing panels
- **Custom Components**: Support for user-defined components

### 🔄 Flow Management
- **Flow CRUD Operations**: Create, read, update, delete flows
- **Version Control**: Track flow changes over time
- **Flow Templates**: Reusable flow patterns
- **Import/Export**: JSON-based flow sharing

### 🚀 Deployment System
- **One-Click Deploy**: Deploy flows to runtime environments
- **Environment Management**: Support for dev/test/prod environments
- **Deployment Tracking**: Monitor deployment status and history
- **Rollback Support**: Revert to previous versions

### 📊 Real-time Monitoring
- **System Metrics**: Live performance and health monitoring
- **Flow Execution**: Real-time flow execution visualization
- **Error Tracking**: Visual error indicators and debugging
- **Resource Usage**: CPU, memory, and queue utilization

### 🌐 Modern Web Interface
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Dark Theme**: Professional dark theme optimized for long usage
- **Keyboard Shortcuts**: Productivity shortcuts for common actions
- **WebSocket Communication**: Real-time updates without page refresh

## Quick Start

### Installation

1. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn websockets pydantic
   ```

2. **Optional LLMFlow Integration**:
   ```bash
   pip install msgpack  # For LLMFlow integration
   ```

### Running the Interface

#### Standalone Mode (No LLMFlow Integration)
```bash
python start_visual_interface.py
```

#### With LLMFlow Integration
```bash
python start_visual_interface.py --llmflow-integration
```

#### Custom Configuration
```bash
python start_visual_interface.py --config visual_interface_config.json
```

#### Development Mode
```bash
python start_visual_interface.py --dev-mode --demo-data
```

### Access the Interface

Open your browser and navigate to:
- **Local**: http://localhost:8080
- **Network**: http://0.0.0.0:8080 (when running with --host 0.0.0.0)

## Configuration

### Creating Configuration File

Generate a sample configuration file:
```bash
python start_visual_interface.py --create-config
```

### Configuration Options

```json
{
  "server": {
    "host": "localhost",
    "port": 8080
  },
  "llmflow": {
    "enabled": true,
    "queue_manager_config": {
      "transport": "udp",
      "host": "localhost", 
      "port": 8421
    },
    "conductor_config": {
      "health_check_interval": 30.0
    }
  },
  "features": {
    "realtime_collaboration": true,
    "auto_save": true,
    "metrics_collection": true
  },
  "ui": {
    "theme": "dark",
    "auto_layout": true,
    "grid_snap": true
  }
}
```

## API Documentation

### REST API Endpoints

#### Flow Management
- `GET /api/flows` - List all flows
- `POST /api/flows` - Create new flow
- `GET /api/flows/{id}` - Get specific flow
- `PUT /api/flows/{id}` - Update flow
- `DELETE /api/flows/{id}` - Delete flow

#### Component Management
- `GET /api/components` - List all components
- `POST /api/components` - Create new component
- `GET /api/components/{id}` - Get specific component

#### Deployment
- `POST /api/deploy` - Deploy flow to runtime
- `GET /api/deployments` - List deployments
- `GET /api/deployments/{id}/status` - Get deployment status

#### Monitoring
- `GET /api/metrics` - Get system metrics
- `GET /api/health` - Health check endpoint

#### Real-time Communication
- `WebSocket /api/realtime` - Real-time updates

### WebSocket Events

#### Client → Server
```javascript
{
  "type": "ping",
  "timestamp": "2025-07-08T10:00:00Z"
}
```

#### Server → Client
```javascript
{
  "type": "flow.updated",
  "timestamp": "2025-07-08T10:00:00Z",
  "data": {
    "flow": { /* flow object */ }
  }
}
```

## Usage Guide

### Creating Your First Flow

1. **Start the Interface**: Run the startup script
2. **Create New Flow**: Click "New Flow" button
3. **Add Components**: Drag components from palette to canvas
4. **Connect Components**: Click and drag between ports
5. **Configure Properties**: Double-click nodes to edit properties
6. **Save Flow**: Click "Save" button
7. **Deploy Flow**: Click "Deploy" button

### Component Types

#### Data Atoms (Blue)
- **EmailAtom**: Email address with validation
- **TimestampAtom**: Timestamp data type
- **PasswordAtom**: Secure password handling

#### Service Atoms (Green)  
- **Validate Email**: Email format validation
- **Hash Password**: Secure password hashing
- **Transform Data**: Data transformation functions

#### Molecules (Orange)
- **Auth Molecule**: Complete authentication workflow
- **Payment Molecule**: Payment processing workflow
- **Validation Molecule**: Data validation workflow

#### Cells (Purple)
- **User Management Cell**: Complete user management system
- **E-commerce Cell**: E-commerce application logic
- **Content Management Cell**: Content management system

#### Infrastructure (Red)
- **Conductor**: Process monitoring and management
- **Queue Manager**: Message queue management
- **Load Balancer**: Traffic distribution

### Keyboard Shortcuts

- **Ctrl+N**: New flow
- **Ctrl+S**: Save flow
- **Ctrl+D**: Deploy flow
- **Delete**: Delete selected node/connection
- **Escape**: Deselect all / Close modals
- **Space**: Pan canvas mode
- **Mouse Wheel**: Zoom in/out

### Tips & Best Practices

1. **Organize Components**: Use consistent naming and grouping
2. **Document Flows**: Add meaningful descriptions to flows and nodes
3. **Test Incrementally**: Deploy and test small changes frequently
4. **Use Templates**: Create reusable flow patterns
5. **Monitor Performance**: Keep an eye on metrics dashboard
6. **Version Control**: Regularly save flow versions

## Development

### Project Structure

```
llmflow/visual/
├── __init__.py              # Package initialization
├── server.py                # FastAPI server and API
└── static/
    ├── css/
    │   └── main.css         # Styling and responsive design
    └── js/
        └── app.js           # Frontend application logic
```

### Extending the Interface

#### Adding Custom Components

1. **Define Component**:
   ```python
   component = ComponentDefinition(
       name="My Custom Component",
       type="atom",
       category="custom",
       description="Custom component description",
       input_types=["InputType"],
       output_types=["OutputType"],
       properties={"setting1": "default_value"},
       icon="custom-icon",
       color="#custom-color"
   )
   ```

2. **Register Component**:
   ```python
   server.components[component.id] = component
   ```

#### Adding Custom API Endpoints

```python
@server.app.get("/api/custom")
async def custom_endpoint():
    return {"message": "Custom API endpoint"}
```

### Testing

Run the test suite:
```bash
python test_phase5_final.py
```

## Troubleshooting

### Common Issues

#### Server Won't Start
- **Check Port**: Ensure port 8080 is available
- **Check Dependencies**: Install required packages
- **Check Config**: Verify configuration file syntax

#### Components Not Loading
- **Check Console**: Open browser developer tools
- **Check Network**: Verify API endpoints are accessible
- **Check WebSocket**: Ensure WebSocket connection is established

#### Flow Won't Deploy
- **Check LLMFlow**: Ensure LLMFlow system is running
- **Check Permissions**: Verify deployment permissions
- **Check Configuration**: Validate flow configuration

### Logs and Debugging

- **Server Logs**: Check `llmflow_visual.log`
- **Browser Console**: Open developer tools → Console
- **Network Tab**: Monitor API requests and responses
- **WebSocket Tab**: Monitor real-time communication

## Contributing

1. **Fork Repository**: Create your own fork
2. **Create Branch**: `git checkout -b feature/new-feature`
3. **Make Changes**: Implement your improvements
4. **Test Changes**: Run test suite
5. **Submit PR**: Create pull request with description

## License

This project is part of the LLMFlow framework and follows the same license terms.

## Support

For questions, issues, or contributions:
- **Documentation**: Check inline help and tooltips
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Community**: Join the LLMFlow community

---

**LLMFlow Visual Interface** - Making queue-based application development visual and intuitive! 🎨✨
