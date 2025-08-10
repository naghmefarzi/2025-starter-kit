"""
Base class for all LLM-based components with consistent error handling.
"""
import logging
from typing import Type, TypeVar, Dict, Any, Optional
from pydantic import BaseModel
from llm_client import llm_client

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger(__name__)


class BaseLLMComponent:
    """Base class for all components that use LLM generation."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.llm_client = llm_client
        logger.info(f"Initialized {component_name}")
    
    def generate_structured_response(
        self,
        response_model: Type[T],
        system_prompt: str,
        user_input: str,
        temperature: Optional[float] = None,
        max_retries: int = 3
    ) -> T:
        """Generate structured response with consistent error handling."""
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input}
        ]
        
        try:
            result = self.llm_client.generate_structured(
                response_model=response_model,
                messages=messages,
                temperature=temperature,
                max_retries=max_retries
            )
            
            logger.debug(f"{self.component_name} generated structured response successfully")
            return result
            
        except Exception as e:
            logger.error(f"{self.component_name} failed to generate response: {e}")
            raise RuntimeError(f"{self.component_name} generation failed: {e}")
    
    def validate_response(self, response: Any, expected_type: Type[T]) -> T:
        """Validate and convert response to expected type."""
        if not isinstance(response, expected_type):
            raise ValueError(f"Expected {expected_type}, got {type(response)}")
        return response