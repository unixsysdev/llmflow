"""
OpenAI LLM Service Atoms

This module provides service atoms for interacting with OpenAI's API
for code analysis, optimization, and intelligent recommendations.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    openai = None
    AsyncOpenAI = None

from ..core.base import ServiceAtom, DataAtom
from ..atoms.data import StringAtom, BooleanAtom, IntegerAtom
from ..plugins.config import get_global_config_loader

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """LLM request data structure."""
    prompt: str
    model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.1
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompt': self.prompt,
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'system_prompt': self.system_prompt,
            'metadata': self.metadata or {}
        }


@dataclass
class LLMResponse:
    """LLM response data structure."""
    content: str
    model: str
    usage_tokens: int
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'model': self.model,
            'usage_tokens': self.usage_tokens,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata or {},
            'error': self.error
        }


class LLMRequestAtom(DataAtom):
    """Data atom for LLM requests."""
    
    def __init__(self, request: LLMRequest):
        super().__init__(request.to_dict())
        self.request = request
    
    def validate(self) -> 'ValidationResult':
        from ..atoms.validation import ValidationResult
        
        errors = []
        
        if not self.request.prompt or not self.request.prompt.strip():
            errors.append("Prompt cannot be empty")
        
        if not self.request.model:
            errors.append("Model must be specified")
        
        if self.request.max_tokens <= 0:
            errors.append("Max tokens must be positive")
        
        if not (0.0 <= self.request.temperature <= 2.0):
            errors.append("Temperature must be between 0.0 and 2.0")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)


class LLMResponseAtom(DataAtom):
    """Data atom for LLM responses."""
    
    def __init__(self, response: LLMResponse):
        super().__init__(response.to_dict())
        self.response = response
    
    def validate(self) -> 'ValidationResult':
        from ..atoms.validation import ValidationResult
        
        errors = []
        
        if self.response.error:
            errors.append(f"LLM error: {self.response.error}")
        
        if not self.response.content:
            errors.append("Response content cannot be empty")
        
        if self.response.usage_tokens < 0:
            errors.append("Usage tokens cannot be negative")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)


class OpenAIServiceAtom(ServiceAtom):
    """Service atom for OpenAI API integration."""
    
    def __init__(self):
        super().__init__(
            name="openai_service",
            input_types=["llmflow.atoms.llm.LLMRequestAtom"],
            output_types=["llmflow.atoms.llm.LLMResponseAtom"]
        )
        
        # Load configuration
        self.config_loader = get_global_config_loader()
        self.llm_config = self.config_loader.get_config('llm') or {}
        
        # Initialize OpenAI client
        self.client = None
        self._initialize_client()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'average_response_time': 0.0
        }
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        if not openai or not AsyncOpenAI:
            logger.error("OpenAI library not installed. Install with: pip install openai>=1.0.0")
            return
        
        # Get API key from environment or config
        api_key = os.getenv('OPENAI_API_KEY') or self.llm_config.get('api_key')
        
        if not api_key:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return
        
        try:
            self.client = AsyncOpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    async def process(self, inputs: List[DataAtom]) -> List[LLMResponseAtom]:
        """Process LLM request."""
        if not inputs or len(inputs) == 0:
            return [LLMResponseAtom(LLMResponse(
                content="",
                model="",
                usage_tokens=0,
                error="No input provided"
            ))]
        
        request_atom = inputs[0]
        
        if not isinstance(request_atom, LLMRequestAtom):
            return [LLMResponseAtom(LLMResponse(
                content="",
                model="",
                usage_tokens=0,
                error="Invalid input type"
            ))]
        
        # Validate request
        validation = request_atom.validate()
        if not validation.is_valid:
            return [LLMResponseAtom(LLMResponse(
                content="",
                model="",
                usage_tokens=0,
                error=f"Validation failed: {validation.errors}"
            ))]
        
        self.stats['total_requests'] += 1
        start_time = datetime.now()
        
        try:
            response = await self._make_openai_request(request_atom.request)
            
            # Update statistics
            response_time = (datetime.now() - start_time).total_seconds()
            self.stats['successful_requests'] += 1
            self.stats['total_tokens_used'] += response.usage_tokens
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (self.stats['successful_requests'] - 1) + response_time) /
                self.stats['successful_requests']
            )
            
            return [LLMResponseAtom(response)]
        
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"OpenAI request failed: {e}")
            
            return [LLMResponseAtom(LLMResponse(
                content="",
                model=request_atom.request.model,
                usage_tokens=0,
                error=str(e)
            ))]
    
    async def _make_openai_request(self, request: LLMRequest) -> LLMResponse:
        """Make request to OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        # Prepare messages
        messages = []
        
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        messages.append({"role": "user", "content": request.prompt})
        
        # Make API call with retry logic
        max_retries = self.llm_config.get('retry_attempts', 3)
        timeout = self.llm_config.get('timeout_seconds', 30)
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=request.model,
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature
                    ),
                    timeout=timeout
                )
                
                # Extract response content
                content = response.choices[0].message.content or ""
                usage_tokens = response.usage.total_tokens if response.usage else 0
                
                # Calculate confidence score based on response length and model
                confidence_score = self._calculate_confidence_score(content, request.model)
                
                return LLMResponse(
                    content=content,
                    model=request.model,
                    usage_tokens=usage_tokens,
                    confidence_score=confidence_score,
                    metadata={
                        'finish_reason': response.choices[0].finish_reason,
                        'created': response.created,
                        'attempt': attempt + 1
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
    
    def _calculate_confidence_score(self, content: str, model: str) -> float:
        """Calculate confidence score for the response."""
        base_score = 0.7
        
        # Adjust based on response length
        if len(content) > 500:
            base_score += 0.1
        elif len(content) < 50:
            base_score -= 0.2
        
        # Adjust based on model
        if model == "gpt-4":
            base_score += 0.1
        elif model == "gpt-3.5-turbo":
            base_score += 0.05
        
        # Check for common error indicators
        error_indicators = ["sorry", "cannot", "unable", "don't know", "unclear"]
        if any(indicator in content.lower() for indicator in error_indicators):
            base_score -= 0.3
        
        return max(0.0, min(1.0, base_score))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            'service': 'openai',
            'stats': self.stats.copy(),
            'config': {
                'model': self.llm_config.get('model', 'gpt-4'),
                'fallback_model': self.llm_config.get('fallback_model', 'gpt-3.5-turbo'),
                'max_tokens': self.llm_config.get('max_tokens', 4000),
                'temperature': self.llm_config.get('temperature', 0.1)
            },
            'client_initialized': self.client is not None,
            'timestamp': datetime.now().isoformat()
        }
