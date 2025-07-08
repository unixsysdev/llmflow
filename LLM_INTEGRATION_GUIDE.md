# LLMFlow OpenAI Integration Guide

🤖 **LLMFlow now includes powerful OpenAI-powered code optimization!**

## 🚀 **What's New**

LLMFlow's meta-network now uses **real LLM intelligence** to analyze code and generate optimization recommendations. Instead of hardcoded templates, the system now:

- ✅ **Analyzes code with GPT-4** for deep performance insights
- ✅ **Generates specific optimizations** tailored to your code
- ✅ **Provides implementation code** with rollback strategies
- ✅ **Validates recommendations** with confidence scoring
- ✅ **Performs system-wide analysis** for architectural improvements

## 🔧 **Setup Instructions**

### 1. Install Dependencies

```bash
# Install OpenAI library (if not already installed)
pip install openai>=1.0.0 aiohttp

# Or install all requirements
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

**Option B: Environment File**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and set your API key
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Verify Installation

```bash
# Run the LLM integration test
python test_llm_integration.py

# Should show:
# 🎉 ALL LLM INTEGRATION TESTS PASSED!
# ✅ OpenAI-powered optimization is ready to use!
```

## 📋 **How It Works**

### **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Conductor     │───▶│  Master Queue   │───▶│   OpenAI API    │
│  (Performance   │    │  (LLM Optimizer)│    │ (GPT-4 Analysis)│
│   Monitoring)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Component     │◀───│ Optimization    │◀───│  Generated      │
│   (Optimized    │    │ Recommendation  │    │  Optimization   │
│    Code)        │    │                 │    │     Code        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **LLM-Powered Molecules**

1. **LLMCodeAnalysisMolecule**
   - Analyzes source code and performance metrics
   - Uses GPT-4 to identify optimization opportunities
   - Returns structured JSON analysis with actionable insights

2. **LLMOptimizationGeneratorMolecule**
   - Generates specific optimization code
   - Provides implementation and rollback strategies
   - Includes risk assessment and validation steps

3. **LLMSystemOptimizationMolecule**
   - Performs system-wide architectural analysis
   - Identifies cross-component optimization opportunities
   - Provides strategic optimization recommendations

## 🎯 **Usage Examples**

### **1. Enable LLM Optimization in Master Queue**

```python
from llmflow.master.optimizer import LLMOptimizer
from llmflow.queue import QueueManager

# Create optimizer with LLM integration
queue_manager = QueueManager()
optimizer = LLMOptimizer(queue_manager)

# Start with LLM optimization enabled
await optimizer.start()

# The optimizer will automatically use LLM analysis
# when components report performance issues
```

### **2. Direct LLM Code Analysis**

```python
from llmflow.molecules.llm_optimization import LLMCodeAnalysisMolecule
from llmflow.atoms.data import StringAtom

# Create analysis molecule
analysis_molecule = LLMCodeAnalysisMolecule(queue_manager)

# Analyze code
source_code = """
def slow_function():
    result = []
    for i in range(10000):
        result.append(i * i)
    return result
"""

metrics = '{"latency_ms": 1500, "error_rate": 0.02}'

# Get LLM analysis
result = await analysis_molecule.process([
    StringAtom(source_code),
    StringAtom(metrics)
])

analysis_report = result[0].value  # Detailed JSON analysis
issues_found = result[1].value     # Boolean: issues detected
```

### **3. Generate Optimizations**

```python
from llmflow.molecules.llm_optimization import LLMOptimizationGeneratorMolecule

# Create optimization generator
optimizer_molecule = LLMOptimizationGeneratorMolecule(queue_manager)

# Generate optimization
result = await optimizer_molecule.process([
    StringAtom(analysis_report),    # From previous analysis
    StringAtom(source_code),        # Original code
    StringAtom("latency_optimization")  # Optimization type
])

recommendation = result[0].recommendation
print(f"Optimization: {recommendation.description}")
print(f"Expected improvement: {recommendation.expected_improvement}")
print(f"Implementation code:\n{recommendation.implementation_code}")
```

## ⚙️ **Configuration Options**

### **LLM Configuration**

```python
# In your configuration
llm_config = {
    'provider': 'openai',
    'model': 'gpt-4',           # Primary model
    'fallback_model': 'gpt-3.5-turbo',  # Fallback model
    'max_tokens': 4000,         # Max response tokens
    'temperature': 0.1,         # Low temperature for consistent results
    'timeout_seconds': 30,      # Request timeout
    'retry_attempts': 3,        # Retry failed requests
    'optimization': {
        'enabled': True,        # Enable LLM optimization
        'confidence_threshold': 0.7,    # Min confidence to apply
        'auto_apply_threshold': 0.9,    # Auto-apply if confidence > 90%
        'validation_required': True     # Require validation before applying
    }
}
```

### **Environment Variables**

```bash
# Core LLM settings
OPENAI_API_KEY=your-api-key
LLMFLOW_LLM_MODEL=gpt-4
LLMFLOW_LLM_TEMPERATURE=0.1

# Optimization settings
LLMFLOW_LLM_OPTIMIZATION_ENABLED=true
LLMFLOW_LLM_CONFIDENCE_THRESHOLD=0.7
LLMFLOW_LLM_AUTO_APPLY_THRESHOLD=0.9
```

## 📊 **Monitoring & Metrics**

### **LLM Optimizer Metrics**

```python
# Get optimizer metrics
metrics = await optimizer.get_optimizer_metrics()

print(f"LLM analyses performed: {metrics['llm_analyses_performed']}")
print(f"LLM recommendations: {metrics['llm_recommendations_generated']}")
print(f"Fallback optimizations: {metrics['fallback_optimizations']}")
print(f"Success rate: {metrics['completed_tasks'] / metrics['total_tasks'] * 100}%")
```

### **OpenAI Service Stats**

```python
from llmflow.atoms.llm import OpenAIServiceAtom

service = OpenAIServiceAtom()
stats = service.get_stats()

print(f"Total requests: {stats['stats']['total_requests']}")
print(f"Successful requests: {stats['stats']['successful_requests']}")
print(f"Total tokens used: {stats['stats']['total_tokens_used']}")
print(f"Average response time: {stats['stats']['average_response_time']:.2f}s")
```

## 🛡️ **Error Handling & Fallbacks**

The system includes robust error handling:

1. **API Failures**: Automatically falls back to traditional optimization
2. **Rate Limits**: Implements exponential backoff and retry logic
3. **Invalid Responses**: Validates LLM responses before applying
4. **Network Issues**: Graceful degradation with local optimization

### **Fallback Configuration**

```python
optimizer_config = {
    'use_llm_optimization': True,    # Primary: Use LLM
    'llm_fallback_enabled': True,    # Fallback: Use traditional optimization
    'max_retries': 3,                # Retry failed LLM requests
    'timeout_seconds': 30            # Request timeout
}
```

## 🔒 **Security Considerations**

1. **API Key Security**: Store API keys securely, never in code
2. **Code Privacy**: Code is sent to OpenAI for analysis
3. **Data Sanitization**: Remove sensitive data before analysis
4. **Access Control**: Limit who can trigger optimizations

### **Secure API Key Management**

```bash
# Use environment variables
export OPENAI_API_KEY="sk-..."

# Or use a secrets management system
# AWS Secrets Manager, Azure Key Vault, etc.
```

## 📈 **Performance Impact**

- **LLM Analysis**: ~2-5 seconds per component analysis
- **Token Usage**: ~1000-3000 tokens per optimization
- **Cost**: Approximately $0.01-0.05 per optimization (GPT-4)
- **Fallback**: <100ms traditional optimization if LLM fails

## 🚀 **Next Steps**

1. **Set up API key** and test the integration
2. **Monitor optimization quality** and adjust confidence thresholds
3. **Review generated optimizations** before auto-applying
4. **Scale up** to system-wide optimization
5. **Monitor costs** and token usage

## 📞 **Support**

- **Documentation**: This guide covers basic usage
- **Examples**: See `test_llm_integration.py` for working examples
- **Issues**: Report bugs or feature requests in the project repo
- **Configuration**: Refer to `.env.example` for all settings

---

🎉 **Congratulations! Your LLMFlow system now has AI-powered self-optimization!** 🤖✨
