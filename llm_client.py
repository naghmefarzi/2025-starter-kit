"""
Safe Ollama client that doesn't test models during initialization.
"""
import logging
import time
import ollama
import requests
from typing import Type, TypeVar, Optional, Dict, Any
from pydantic import BaseModel
from config import CONFIG
import json

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger(__name__)


class SafeLLMClient:
    """Safe Ollama client that defers model testing until actual use."""
    
    def __init__(self):
        self.model = None
        self.available_models = []
        self.tested_models = set()
        self.working_model = None
        self._initialize_connection_only()
    
    def _initialize_connection_only(self):
        """Initialize connection to Ollama without testing models."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{CONFIG.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise ConnectionError("Ollama server not responding")
            
            # Get available models
            models_response = requests.get(f"{CONFIG.base_url}/api/tags")
            self.available_models = [model['name'] for model in models_response.json().get('models', [])]
            
            logger.info(f"Connected to Ollama. Available models: {self.available_models}")
            
            # Create client connection but don't test any models yet
            self.model = ollama.Client(host=CONFIG.base_url)
            
            logger.info("Ollama client initialized successfully (no models tested yet)")
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def _find_working_model(self):
        """Only test the configured model from CONFIG.model_name."""
        if self.working_model:
            return self.working_model

        if not CONFIG.model_name:
            raise RuntimeError("MODEL_NAME is not set in your .env file.")

        model_to_try = CONFIG.model_name.strip()
        logger.info(f"Testing configured model: {model_to_try}")

        try:
            # Minimal test to confirm model runs
            response = self.model.chat(
                model=model_to_try,
                messages=[{"role": "user", "content": "Hi"}],
                options={
                    'temperature': 0.0,
                    'max_tokens': 3,
                    'top_p': 1.0
                }
            )
            logger.info(f"✓ Model {model_to_try} works!")
            self.working_model = model_to_try
            return model_to_try

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Configured model {model_to_try} failed: {error_msg}")
            if "memory" in error_msg.lower() or "broken pipe" in error_msg.lower():
                raise RuntimeError(
                    f"Configured model {model_to_try} is too large for your system. "
                    "Please choose a smaller one in MODEL_NAME."
                )
            raise

    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks if present."""
        import re
        
        content = content.strip()
        
        # Look for ```json ... ``` blocks anywhere in the text
        json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Look for ``` ... ``` blocks anywhere in the text
        code_match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            # Check if the content looks like JSON (starts with { or [)
            potential_json = code_match.group(1).strip()
            if potential_json.startswith(('{', '[')):
                return potential_json
        
        # Look for JSON objects that start with { and end with } (even without markdown)
        json_object_match = re.search(r'(\{.*\})', content, re.DOTALL)
        if json_object_match:
            return json_object_match.group(1).strip()
        
        # Return as-is if no JSON detected
        return content

    def generate_structured(
                            self,
                            response_model: Type[BaseModel],
                            messages: list[Dict[str, str]],
                            temperature: Optional[float] = None,
                            max_retries: int = 3
                        ) -> BaseModel:
        """Generate structured output using Ollama."""
        if not self.model:
            raise RuntimeError("Model not initialized")

        temp = temperature if temperature is not None else CONFIG.temperature
        
        for attempt in range(max_retries):
            try:
                if CONFIG.debug_mode:
                    logging.debug(f"Generating structured output (attempt {attempt + 1})")
                    logging.debug(f"Messages: {messages[:200]}...")
                
                # Call Ollama's chat API with num_predict instead of max_tokens
                response = self.model.chat(
                    model=CONFIG.model_name,
                    messages=messages,
                    options={'temperature': temp, 'num_predict': CONFIG.max_tokens}
                )
                
                # Extract response content
                response_content = response['message']['content'].strip()
                if CONFIG.debug_mode:
                    logging.debug(f"Raw response: {response_content}")
                
                # Clean up markdown code blocks if present
                cleaned_content = self._extract_json_from_markdown(response_content)
                if CONFIG.debug_mode and cleaned_content != response_content:
                    logging.debug(f"Cleaned response: {cleaned_content}")
                
                # Try to parse as JSON
                try:
                    json_response = json.loads(cleaned_content)
                    if CONFIG.debug_mode:
                        logging.debug(f"Parsed JSON: {json_response}")
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse JSON on attempt {attempt + 1}: {e}")
                    logging.warning(f"Cleaned response was: {cleaned_content}")
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"LLM failed to generate valid JSON after {max_retries} attempts. Last response: {cleaned_content}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                # Parse with Pydantic (use model_validate instead of deprecated parse_obj)
                try:
                    return response_model.model_validate(json_response)
                except Exception as e:
                    logging.warning(f"Pydantic validation failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to validate response with Pydantic after {max_retries} attempts: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
            except Exception as e:
                logging.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to generate after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def health_check(self) -> bool:
        """Check if we can connect and have at least one working model."""
        try:
            # Check server connection
            response = requests.get(f"{CONFIG.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Try to find at least one working model
            if not self.working_model:
                try:
                    self._find_working_model()
                    return True
                except Exception:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_working_model(self):
        """Get the name of the currently working model."""
        if not self.working_model:
            self._find_working_model()
        return self.working_model


# Global LLM client instance
llm_client = SafeLLMClient()