"""
OpenRouter LLM Service Atoms

Enhanced OpenRouter integration for LLMFlow with support for multiple AI models
including Gemini 2.0 Flash and other cutting-edge models.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

try:
    from ..core.base import ServiceAtom, DataAtom, ValidationResult
except ImportError:
    # Fallback for development
    ServiceAtom = object
    DataAtom = object
    ValidationResult = object
from ..atoms.data import StringAtom, BooleanAtom, IntegerAtom
try:
    from ..plugins.config import get_global_config_loader
except ImportError:
    # Fallback for development
    def get_global_config_loader():
        class MockLoader:
            def get_config(self, key):
                return {}
        return MockLoader()

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterRequest:
    """OpenRouter LLM request data structure."""
    prompt: str
    model: str = "google/gemini-2.0-flash-001"
    max_tokens: int = 4000
    temperature: float = 0.1
    system_prompt: Optional[str] = None
    site_url: Optional[str] = None
    site_name: Optional[str] = None
    image_urls: List[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompt': self.prompt,
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'system_prompt': self.system_prompt,
            'site_url': self.site_url,
            'site_name': self.site_name,
            'image_urls': self.image_urls or [],
            'metadata': self.metadata or {}
        }


@dataclass
class OpenRouterResponse:
    """OpenRouter LLM response data structure."""
    content: str
    model: str
    usage_tokens: int
    cost_usd: float = 0.0
    confidence_score: float = 0.0
    provider: str = "openrouter"
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'model': self.model,
            'usage_tokens': self.usage_tokens,
            'cost_usd': self.cost_usd,
            'confidence_score': self.confidence_score,
            'provider': self.provider,
            'metadata': self.metadata or {},
            'error': self.error
        }


class OpenRouterRequestAtom(DataAtom):
    """Data atom for OpenRouter requests."""
    
    def __init__(self, request: OpenRouterRequest):
        super().__init__(request.to_dict())
        self.request = request
    
    def validate(self) -> ValidationResult:
        """Validate the OpenRouter request."""
        errors = []
        
        if not self.request.prompt or not self.request.prompt.strip():
            errors.append("Prompt cannot be empty")
        
        if not self.request.model:
            errors.append("Model must be specified")
        
        if self.request.max_tokens <= 0:
            errors.append("Max tokens must be positive")
        
        if not (0.0 <= self.request.temperature <= 2.0):
            errors.append("Temperature must be between 0.0 and 2.0")
            
        # Validate image URLs if provided
        if self.request.image_urls:
            for url in self.request.image_urls:
                if not url.startswith(('http://', 'https://')):
                    errors.append(f"Invalid image URL: {url}")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def serialize(self) -> bytes:
        """Serialize the request to bytes."""
        import json
        return json.dumps(self.value).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'OpenRouterRequestAtom':
        """Deserialize bytes to request atom."""
        import json
        request_data = json.loads(data.decode('utf-8'))
        request = OpenRouterRequest(**request_data)
        return cls(request)


class OpenRouterResponseAtom(DataAtom):
    """Data atom for OpenRouter responses."""
    
    def __init__(self, response: OpenRouterResponse):
        super().__init__(response.to_dict())
        self.response = response
    
    def validate(self) -> ValidationResult:
        """Validate the OpenRouter response."""
        errors = []
        
        if self.response.error:
            errors.append(f"OpenRouter error: {self.response.error}")
        
        if not self.response.content and not self.response.error:
            errors.append("Response content cannot be empty")
        
        if self.response.usage_tokens < 0:
            errors.append("Usage tokens cannot be negative")
            
        if self.response.cost_usd < 0:
            errors.append("Cost cannot be negative")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def serialize(self) -> bytes:
        """Serialize the response to bytes."""
        import json
        return json.dumps(self.value).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'OpenRouterResponseAtom':
        """Deserialize bytes to response atom."""
        import json
        response_data = json.loads(data.decode('utf-8'))
        response = OpenRouterResponse(**response_data)
        return cls(response)


class OpenRouterServiceAtom(ServiceAtom):
    """Enhanced OpenRouter service atom for LLM optimization."""
    
    # Supported models with their capabilities
    SUPPORTED_MODELS = {
        "google/gemini-2.0-flash-001": {
            "max_tokens": 8192,
            "supports_vision": True,
            "cost_per_1k_tokens": 0.075,
            "optimization_score": 0.95
        },
        "anthropic/claude-3.5-sonnet": {
            "max_tokens": 8192,
            "supports_vision": True,
            "cost_per_1k_tokens": 0.15,
            "optimization_score": 0.92
        },
        "openai/gpt-4": {
            "max_tokens": 8192,
            "supports_vision": False,
            "cost_per_1k_tokens": 0.30,
            "optimization_score": 0.90
        },
        "meta-llama/llama-3.1-405b-instruct": {
            "max_tokens": 4096,
            "supports_vision": False,
            "cost_per_1k_tokens": 0.27,
            "optimization_score": 0.88
        }
    }
    
    def __init__(self):
        super().__init__(
            name="openrouter_service",
            input_types=["llmflow.atoms.openrouter_llm.OpenRouterRequestAtom"],
            output_types=["llmflow.atoms.openrouter_llm.OpenRouterResponseAtom"]
        )
        
        # Load configuration
        self.config_loader = get_global_config_loader()
        self.openrouter_config = self.config_loader.get_config('openrouter') or {}
        
        # Initialize OpenRouter client
        self.client = None
        self._initialize_client()
        
        # Enhanced statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_cost_usd': 0.0,
            'average_response_time': 0.0,
            'model_usage': {},
            'optimization_requests': 0,
            'code_generation_requests': 0
        }
        
        # Optimization context
        self.optimization_context = {
            'active_optimizations': {},
            'performance_baseline': {},
            'learning_history': []
        }
    
    def _initialize_client(self) -> None:
        """Initialize OpenRouter client."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI library not installed. Install with: pip install openai>=1.0.0")
            return
        
        # Get API key from environment or config
        api_key = os.getenv('OPENROUTER_API_KEY') or self.openrouter_config.get('api_key')
        
        if not api_key:
            logger.warning("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
            return
        
        try:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            logger.info("OpenRouter client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
    
    async def process(self, inputs: List[DataAtom]) -> List[OpenRouterResponseAtom]:
        """Process OpenRouter LLM request with optimization context."""
        if not inputs or len(inputs) == 0:
            return [OpenRouterResponseAtom(OpenRouterResponse(
                content="",
                model="",
                usage_tokens=0,
                error="No input provided"
            ))]
        
        request_atom = inputs[0]
        
        if not isinstance(request_atom, OpenRouterRequestAtom):
            return [OpenRouterResponseAtom(OpenRouterResponse(
                content="",
                model="",
                usage_tokens=0,
                error="Invalid input type"
            ))]
        
        # Validate request
        validation = request_atom.validate()
        if not validation.is_valid:
            return [OpenRouterResponseAtom(OpenRouterResponse(
                content="",
                model="",
                usage_tokens=0,
                error=f"Validation failed: {validation.errors}"
            ))]
        
        # Auto-select optimal model if requested
        if request_atom.request.model == "auto":
            request_atom.request.model = self._select_optimal_model(request_atom.request)
        
        self.stats['total_requests'] += 1
        start_time = datetime.now()
        
        # Track request type
        if 'optimization' in request_atom.request.prompt.lower():
            self.stats['optimization_requests'] += 1
        if 'generate' in request_atom.request.prompt.lower() or 'implement' in request_atom.request.prompt.lower():
            self.stats['code_generation_requests'] += 1
        
        try:
            response = await self._make_openrouter_request(request_atom.request)
            
            # Update statistics
            response_time = (datetime.now() - start_time).total_seconds()
            self.stats['successful_requests'] += 1
            self.stats['total_tokens_used'] += response.usage_tokens
            self.stats['total_cost_usd'] += response.cost_usd
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (self.stats['successful_requests'] - 1) + response_time) /
                self.stats['successful_requests']
            )
            
            # Track model usage
            model = response.model
            if model not in self.stats['model_usage']:
                self.stats['model_usage'][model] = 0
            self.stats['model_usage'][model] += 1
            
            # Store optimization context if this was an optimization request
            if 'optimization' in request_atom.request.prompt.lower():
                await self._update_optimization_context(request_atom.request, response)
            
            return [OpenRouterResponseAtom(response)]
        
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"OpenRouter request failed: {e}")
            
            return [OpenRouterResponseAtom(OpenRouterResponse(
                content="",
                model=request_atom.request.model,
                usage_tokens=0,
                error=str(e)
            ))]
    
    def _select_optimal_model(self, request: OpenRouterRequest) -> str:
        """Automatically select the optimal model based on request characteristics."""
        # Analyze request complexity
        prompt_length = len(request.prompt)
        has_images = bool(request.image_urls)
        is_optimization = 'optimization' in request.prompt.lower()
        is_code_generation = any(keyword in request.prompt.lower() for keyword in ['generate', 'implement', 'create', 'build'])
        
        # Score models based on requirements
        model_scores = {}
        
        for model, info in self.SUPPORTED_MODELS.items():
            score = info['optimization_score']
            
            # Prefer vision models for image inputs
            if has_images and info['supports_vision']:
                score += 0.1
            elif has_images and not info['supports_vision']:
                score -= 0.5
            
            # Prefer high-performance models for optimization tasks
            if is_optimization:
                score += 0.05
            
            # Consider cost efficiency for longer prompts
            if prompt_length > 2000:
                score -= info['cost_per_1k_tokens'] / 100
            
            # Prefer models with sufficient context for code generation
            if is_code_generation and info['max_tokens'] >= 8192:
                score += 0.05
            
            model_scores[model] = score
        
        # Select the highest scoring model
        optimal_model = max(model_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Auto-selected model {optimal_model} for request (score: {model_scores[optimal_model]:.3f})")
        
        return optimal_model
    
    async def _make_openrouter_request(self, request: OpenRouterRequest) -> OpenRouterResponse:
        """Make request to OpenRouter API."""
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized")
        
        # Prepare messages
        messages = []
        
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        # Handle multimodal content (text + images)
        if request.image_urls:
            content = [{"type": "text", "text": request.prompt}]
            for image_url in request.image_urls:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": request.prompt})
        
        # Prepare extra headers
        extra_headers = {}
        if request.site_url:
            extra_headers["HTTP-Referer"] = request.site_url
        if request.site_name:
            extra_headers["X-Title"] = request.site_name
        
        # Make API call with retry logic
        max_retries = self.openrouter_config.get('retry_attempts', 3)
        timeout = self.openrouter_config.get('timeout_seconds', 45)
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=request.model,
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        extra_headers=extra_headers
                    ),
                    timeout=timeout
                )
                
                # Extract response content
                content = response.choices[0].message.content or ""
                usage_tokens = response.usage.total_tokens if response.usage else 0
                
                # Calculate cost based on model
                model_info = self.SUPPORTED_MODELS.get(request.model, {})
                cost_usd = (usage_tokens / 1000.0) * model_info.get('cost_per_1k_tokens', 0.1)
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(content, request.model, response)
                
                return OpenRouterResponse(
                    content=content,
                    model=request.model,
                    usage_tokens=usage_tokens,
                    cost_usd=cost_usd,
                    confidence_score=confidence_score,
                    provider="openrouter",
                    metadata={
                        'finish_reason': response.choices[0].finish_reason,
                        'created': response.created,
                        'attempt': attempt + 1,
                        'model_info': model_info,
                        'auto_selected': request.model == "auto"
                    }
                )
            
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise RuntimeError(f"Request timed out after {max_retries} attempts")
            
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise e
    
    def _calculate_confidence_score(self, content: str, model: str, response) -> float:
        """Calculate confidence score for the response."""
        base_score = 0.7
        
        # Adjust based on response length and structure
        if len(content) > 500:
            base_score += 0.1
        elif len(content) < 50:
            base_score -= 0.2
        
        # Adjust based on model capability
        model_info = self.SUPPORTED_MODELS.get(model, {})
        base_score += (model_info.get('optimization_score', 0.7) - 0.7) * 0.5
        
        # Check for structured output (JSON, code blocks)
        if '```' in content or (content.strip().startswith('{') and content.strip().endswith('}')):
            base_score += 0.1
        
        # Check for common error indicators
        error_indicators = ["sorry", "cannot", "unable", "don't know", "unclear", "error"]
        if any(indicator in content.lower() for indicator in error_indicators):
            base_score -= 0.3
        
        # Check for optimization-specific indicators
        if any(keyword in content.lower() for keyword in ['optimization', 'improvement', 'performance', 'efficiency']):
            base_score += 0.05
        
        # Adjust based on finish reason
        if hasattr(response.choices[0], 'finish_reason'):
            if response.choices[0].finish_reason == 'stop':
                base_score += 0.1
            elif response.choices[0].finish_reason == 'length':
                base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    async def _update_optimization_context(self, request: OpenRouterRequest, response: OpenRouterResponse):
        """Update optimization context based on the request/response."""
        try:
            context_key = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.optimization_context['active_optimizations'][context_key] = {
                'request': request.to_dict(),
                'response': response.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'confidence': response.confidence_score,
                'cost': response.cost_usd
            }
            
            # Keep only the last 100 optimization contexts
            if len(self.optimization_context['active_optimizations']) > 100:
                oldest_key = min(self.optimization_context['active_optimizations'].keys())
                del self.optimization_context['active_optimizations'][oldest_key]
            
            # Learn from optimization patterns
            self.optimization_context['learning_history'].append({
                'model_used': response.model,
                'confidence_achieved': response.confidence_score,
                'cost_efficiency': response.usage_tokens / max(response.cost_usd, 0.001),
                'request_type': 'optimization',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.warning(f"Failed to update optimization context: {e}")
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization context."""
        insights = {
            'total_optimizations': len(self.optimization_context['active_optimizations']),
            'average_confidence': 0.0,
            'cost_efficiency': 0.0,
            'preferred_models': {},
            'optimization_trends': []
        }
        
        if self.optimization_context['learning_history']:
            history = self.optimization_context['learning_history']
            
            # Calculate average confidence
            insights['average_confidence'] = sum(h['confidence_achieved'] for h in history) / len(history)
            
            # Calculate cost efficiency
            insights['cost_efficiency'] = sum(h['cost_efficiency'] for h in history) / len(history)
            
            # Identify preferred models
            model_usage = {}
            for h in history:
                model = h['model_used']
                if model not in model_usage:
                    model_usage[model] = {'count': 0, 'avg_confidence': 0.0}
                model_usage[model]['count'] += 1
                model_usage[model]['avg_confidence'] += h['confidence_achieved']
            
            for model, data in model_usage.items():
                data['avg_confidence'] /= data['count']
            
            insights['preferred_models'] = model_usage
        
        return insights
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        return {
            'service': 'openrouter',
            'stats': self.stats.copy(),
            'config': {
                'supported_models': list(self.SUPPORTED_MODELS.keys()),
                'default_model': self.openrouter_config.get('default_model', 'google/gemini-2.0-flash-001'),
                'retry_attempts': self.openrouter_config.get('retry_attempts', 3),
                'timeout_seconds': self.openrouter_config.get('timeout_seconds', 45)
            },
            'optimization_insights': self.get_optimization_insights(),
            'client_initialized': self.client is not None,
            'timestamp': datetime.now().isoformat()
        }
