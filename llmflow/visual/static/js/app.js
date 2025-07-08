/**
 * LLMFlow Visual Interface JavaScript Application
 * 
 * This file contains the main client-side application logic for the
 * LLMFlow visual flow designer interface.
 */

class LLMFlowApp {
    constructor() {
        this.currentFlow = null;
        this.selectedNode = null;
        this.draggedNode = null;
        this.connections = [];
        this.nodes = [];
        this.components = [];
        this.websocket = null;
        this.isConnecting = false;
        this.canvasOffset = { x: 0, y: 0 };
        this.canvasScale = 1;
        this.isDraggingCanvas = false;
        this.lastMousePos = { x: 0, y: 0 };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupWebSocket();
        this.loadComponents();
        this.setupDragAndDrop();
        this.setupCanvasInteraction();
        this.initializeTabs();
        this.loadFlows();
        this.startMetricsUpdate();
    }
    
    setupEventListeners() {
        // Header buttons
        document.getElementById('new-flow').addEventListener('click', () => this.showNewFlowModal());
        document.getElementById('save-flow').addEventListener('click', () => this.saveCurrentFlow());
        document.getElementById('deploy-flow').addEventListener('click', () => this.deployCurrentFlow());
        document.getElementById('toggle-monitoring').addEventListener('click', () => this.toggleMonitoring());
        
        // Modal events
        document.getElementById('save-flow-btn').addEventListener('click', () => this.createNewFlow());
        
        // Canvas controls
        document.getElementById('zoom-in').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoom-out').addEventListener('click', () => this.zoomOut());
        document.getElementById('zoom-fit').addEventListener('click', () => this.zoomToFit());
        document.getElementById('center-view').addEventListener('click', () => this.centerView());
        
        // Properties panel
        document.getElementById('close-properties').addEventListener('click', () => this.closePropertiesPanel());
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // Search functionality
        document.getElementById('component-search').addEventListener('input', (e) => this.filterComponents(e.target.value));
        document.getElementById('flow-search').addEventListener('input', (e) => this.filterFlows(e.target.value));
        
        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
        
        // Canvas mouse events
        const canvas = document.getElementById('flow-canvas');
        canvas.addEventListener('mousedown', (e) => this.handleCanvasMouseDown(e));
        canvas.addEventListener('mousemove', (e) => this.handleCanvasMouseMove(e));
        canvas.addEventListener('mouseup', (e) => this.handleCanvasMouseUp(e));
        canvas.addEventListener('wheel', (e) => this.handleCanvasWheel(e));
        canvas.addEventListener('contextmenu', (e) => this.handleCanvasRightClick(e));
    }
    
    setupWebSocket() {
        if (this.isConnecting) return;
        
        this.isConnecting = true;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/realtime`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
            this.isConnecting = false;
        };
        
        this.websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            this.isConnecting = false;
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.setupWebSocket(), 3000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.isConnecting = false;
        };
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'flow.created':
            case 'flow.updated':
                this.handleFlowUpdate(message.data);
                break;
            case 'flow.deleted':
                this.handleFlowDeleted(message.data.flow_id);
                break;
            case 'component.created':
                this.handleComponentCreated(message.data);
                break;
            case 'deployment.started':
            case 'deployment.completed':
                this.handleDeploymentUpdate(message.data);
                break;
            case 'metrics.update':
                this.updateMetrics(message.data);
                break;
        }
    }
    
    updateConnectionStatus(connected) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-indicator span');
        
        if (connected) {
            statusDot.className = 'status-dot status-connected';
            statusText.textContent = 'Connected';
        } else {
            statusDot.className = 'status-dot status-disconnected';
            statusText.textContent = 'Disconnected';
        }
    }
    
    async loadComponents() {
        try {
            const response = await fetch('/api/components');
            const components = await response.json();
            this.components = components;
            this.renderComponentPalette();
        } catch (error) {
            console.error('Failed to load components:', error);
            this.showError('Failed to load components');
        }
    }
    
    renderComponentPalette() {
        const palette = document.getElementById('component-palette');
        const categories = this.groupComponentsByCategory();
        
        palette.innerHTML = '';
        
        Object.entries(categories).forEach(([category, components]) => {
            const categoryElement = this.createCategoryElement(category, components);
            palette.appendChild(categoryElement);
        });
    }
    
    groupComponentsByCategory() {
        const categories = {};
        this.components.forEach(component => {
            const category = component.category || 'Other';
            if (!categories[category]) {
                categories[category] = [];
            }
            categories[category].push(component);
        });
        return categories;
    }
    
    createCategoryElement(categoryName, components) {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'component-category';
        
        const header = document.createElement('div');
        header.className = 'category-header';
        header.innerHTML = `
            <span>${this.capitalizeFirst(categoryName)}</span>
            <i class="fas fa-chevron-down"></i>
        `;
        
        const content = document.createElement('div');
        content.className = 'category-content expanded';
        
        components.forEach(component => {
            const item = this.createComponentItem(component);
            content.appendChild(item);
        });
        
        header.addEventListener('click', () => {
            content.classList.toggle('expanded');
            const icon = header.querySelector('i');
            icon.classList.toggle('fa-chevron-down');
            icon.classList.toggle('fa-chevron-right');
        });
        
        categoryDiv.appendChild(header);
        categoryDiv.appendChild(content);
        
        return categoryDiv;
    }
    
    createComponentItem(component) {
        const item = document.createElement('div');
        item.className = 'component-item';
        item.draggable = true;
        item.dataset.componentId = component.id;
        
        item.innerHTML = `
            <div class="component-icon" style="background-color: ${component.color}">
                <i class="fas fa-${component.icon}"></i>
            </div>
            <div class="component-info">
                <h4>${component.name}</h4>
                <p>${component.description}</p>
            </div>
        `;
        
        return item;
    }
    
    setupDragAndDrop() {
        const palette = document.getElementById('component-palette');
        
        palette.addEventListener('dragstart', (e) => {
            if (e.target.closest('.component-item')) {
                const componentId = e.target.closest('.component-item').dataset.componentId;
                e.dataTransfer.setData('text/plain', componentId);
                e.dataTransfer.effectAllowed = 'copy';
            }
        });
        
        const canvas = document.getElementById('flow-canvas');
        
        canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
        });
        
        canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            const componentId = e.dataTransfer.getData('text/plain');
            if (componentId) {
                const canvasRect = canvas.getBoundingClientRect();
                const x = (e.clientX - canvasRect.left - this.canvasOffset.x) / this.canvasScale;
                const y = (e.clientY - canvasRect.top - this.canvasOffset.y) / this.canvasScale;
                this.addNodeToCanvas(componentId, x, y);
            }
        });
    }
    
    addNodeToCanvas(componentId, x, y) {
        const component = this.components.find(c => c.id === componentId);
        if (!component) return;
        
        const nodeId = this.generateId();
        const node = {
            id: nodeId,
            componentId: componentId,
            x: x,
            y: y,
            component: component,
            properties: { ...component.properties }
        };
        
        this.nodes.push(node);
        this.renderNode(node);
        this.selectNode(nodeId);
    }
    
    renderNode(node) {
        const nodesLayer = document.getElementById('nodes-layer');
        const nodeElement = document.createElement('div');
        nodeElement.className = `flow-node node-${node.component.type}`;
        nodeElement.id = `node-${node.id}`;
        nodeElement.style.left = `${node.x}px`;
        nodeElement.style.top = `${node.y}px`;
        
        nodeElement.innerHTML = `
            <div class="node-header">
                <div class="node-icon" style="background-color: ${node.component.color}">
                    <i class="fas fa-${node.component.icon}"></i>
                </div>
                <div class="node-title">${node.component.name}</div>
            </div>
            <div class="node-body">
                <div class="node-description">${node.component.description}</div>
                <div class="node-ports">
                    <div class="input-ports">
                        ${node.component.input_types.map((type, i) => 
                            `<div class="port input" data-port-id="input-${i}" data-port-type="${type}"></div>`
                        ).join('')}
                    </div>
                    <div class="output-ports">
                        ${node.component.output_types.map((type, i) => 
                            `<div class="port output" data-port-id="output-${i}" data-port-type="${type}"></div>`
                        ).join('')}
                    </div>
                </div>
            </div>
        `;
        
        // Add event listeners for node interaction
        this.setupNodeEvents(nodeElement, node);
        
        nodesLayer.appendChild(nodeElement);
    }
    
    generateId() {
        return 'id-' + Math.random().toString(36).substr(2, 9);
    }
    
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
    
    showError(message) {
        console.error(message);
        // TODO: Implement proper error display
    }
    
    showNewFlowModal() {
        const modal = document.getElementById('flow-modal');
        modal.classList.add('show');
    }
    
    async createNewFlow() {
        // TODO: Implement flow creation
        console.log('Create new flow');
    }
    
    async saveCurrentFlow() {
        // TODO: Implement flow saving
        console.log('Save current flow');
    }
    
    async deployCurrentFlow() {
        // TODO: Implement flow deployment
        console.log('Deploy current flow');
    }
    
    toggleMonitoring() {
        this.switchTab('monitoring');
    }
    
    zoomIn() {
        this.canvasScale = Math.min(this.canvasScale * 1.2, 3);
        this.updateCanvasTransform();
    }
    
    zoomOut() {
        this.canvasScale = Math.max(this.canvasScale / 1.2, 0.1);
        this.updateCanvasTransform();
    }
    
    zoomToFit() {
        // TODO: Implement zoom to fit
        console.log('Zoom to fit');
    }
    
    centerView() {
        this.canvasOffset = { x: 0, y: 0 };
        this.canvasScale = 1;
        this.updateCanvasTransform();
    }
    
    closePropertiesPanel() {
        const panel = document.getElementById('properties-panel');
        panel.classList.remove('open');
    }
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }
    
    filterComponents(searchTerm) {
        // TODO: Implement component filtering
        console.log('Filter components:', searchTerm);
    }
    
    filterFlows(searchTerm) {
        // TODO: Implement flow filtering
        console.log('Filter flows:', searchTerm);
    }
    
    handleKeyboard(e) {
        // TODO: Implement keyboard shortcuts
        console.log('Keyboard event:', e.key);
    }
    
    setupCanvasInteraction() {
        // TODO: Implement canvas interaction
        console.log('Setup canvas interaction');
    }
    
    handleCanvasMouseDown(e) {
        // TODO: Implement canvas mouse down
        console.log('Canvas mouse down');
    }
    
    handleCanvasMouseMove(e) {
        // TODO: Implement canvas mouse move
        console.log('Canvas mouse move');
    }
    
    handleCanvasMouseUp(e) {
        // TODO: Implement canvas mouse up
        console.log('Canvas mouse up');
    }
    
    handleCanvasWheel(e) {
        // TODO: Implement canvas wheel
        console.log('Canvas wheel');
    }
    
    handleCanvasRightClick(e) {
        e.preventDefault();
        // TODO: Implement context menu
        console.log('Canvas right click');
    }
    
    initializeTabs() {
        this.switchTab('components');
    }
    
    async loadFlows() {
        // TODO: Implement flow loading
        console.log('Load flows');
    }
    
    async startMetricsUpdate() {
        // TODO: Implement metrics update
        console.log('Start metrics update');
    }
    
    setupNodeEvents(nodeElement, node) {
        // Node selection
        nodeElement.addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectNode(node.id);
        });
        
        // Node dragging
        let isDragging = false;
        let dragStart = { x: 0, y: 0 };
        
        nodeElement.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return; // Only left mouse button
            
            isDragging = true;
            dragStart = { x: e.clientX - node.x, y: e.clientY - node.y };
            nodeElement.classList.add('dragging');
            
            const onMouseMove = (e) => {
                if (!isDragging) return;
                
                node.x = (e.clientX - dragStart.x);
                node.y = (e.clientY - dragStart.y);
                
                nodeElement.style.left = `${node.x}px`;
                nodeElement.style.top = `${node.y}px`;
            };
            
            const onMouseUp = () => {
                isDragging = false;
                nodeElement.classList.remove('dragging');
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
            };
            
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
    }
    
    createConnection(fromNode, fromPort, fromType, toNode, toPort, toType) {
        const connection = {
            id: this.generateId(),
            from: { node: fromNode, port: fromPort, type: fromType },
            to: { node: toNode, port: toPort, type: toType }
        };
        
        this.connections.push(connection);
        console.log('Connection created:', connection);
        return connection;
    }
    
    selectNode(nodeId) {
        // TODO: Implement node selection
        console.log('Select node:', nodeId);
    }
    
    updateCanvasTransform() {
        // TODO: Implement canvas transform update
        console.log('Update canvas transform');
    }
    
    handleFlowUpdate(data) {
        console.log('Flow update:', data);
    }
    
    handleFlowDeleted(flowId) {
        console.log('Flow deleted:', flowId);
    }
    
    handleComponentCreated(data) {
        console.log('Component created:', data);
    }
    
    handleDeploymentUpdate(data) {
        console.log('Deployment update:', data);
    }
    
    updateMetrics(data) {
        console.log('Metrics update:', data);
    }
}

// Global functions for modal interaction
window.closeModal = function() {
    document.getElementById('flow-modal').classList.remove('show');
    document.getElementById('flow-name').value = '';
    document.getElementById('flow-description').value = '';
};

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.llmflowApp = new LLMFlowApp();
});
