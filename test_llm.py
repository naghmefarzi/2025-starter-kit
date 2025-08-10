#!/usr/bin/env python3
"""
Test script to verify LLM backend setup and functionality.
"""
import json
import logging
from pydantic import BaseModel
from config import CONFIG
from llm_client import SafeLLMClient
from modules.query_generator import QueryGenerator  

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestResponse(BaseModel):
    message: str
    backend: str
    model: str

def test_basic_generation():
    """Test basic structured generation."""
    print("Testing basic LLM generation...")
    
    try:
        llm_client = SafeLLMClient()  # Create new instance
        response = llm_client.generate_structured(
            response_model=TestResponse,
            messages=[
                {
                    'role': 'system', 
                    'content': 'You are a helpful assistant. Respond with a JSON object containing a message, backend name, and model name.'
                },
                {
                    'role': 'user', 
                    'content': f'Hello! Please confirm you are working with Ollama and model: {CONFIG.model_name}'
                }
            ],
            temperature=0.1
        )
        
        print("YAY! Basic generation test passed!")
        print(f"Response: {response.message}")
        print(f"Backend: {response.backend}")
        print(f"Model: {response.model}")
        return True
        
    except Exception as e:
        print(f"X Basic generation test failed: {e}")
        return False

def test_query_generation():
    """Test query generation component."""
    print("\nTesting QueryGenerator component...")
    
    try:
        query_gen = QueryGenerator()
        # Simple test article
        test_article = json.dumps({
            "url": "https://example.com/test",
            "title": "Test Article",
            "headings": ["Introduction"],
            "body": "This is a test article about climate change research findings."
        }, indent=4)
        
        queries = query_gen.generate_query(test_article)
        
        print(f"Yay! QueryGenerator test passed! Generated {len(queries)} queries")
        for i, (query, rationale) in enumerate(queries[:2], 1):  # Show first 2
            print(f"Query {i}: {query}")
            print(f"Rationale: {rationale[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"X QueryGenerator test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("TREC DRAGUN System - LLM Backend Test")
    print("=" * 40)
    print(f"Model: {CONFIG.model_name}")
    print(f"Base URL: {CONFIG.base_url}")
    print("=" * 40)
    
    llm_client = SafeLLMClient()
    print("Performing health check...")
    if not llm_client.health_check():
        print("X Health check failed! Please ensure your LLM backend is running.")
        return
    
    print("Yay! Health check passed!")
    
    tests_passed = 0
    total_tests = 2
    
    if test_basic_generation():
        tests_passed += 1
    
    if test_query_generation():
        tests_passed += 1
    
    print(f"\n{'='*40}")
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("Yay! All tests passed! Your LLM backend is ready.")
    else:
        print("X Some tests failed. Please check your configuration.")

if __name__ == "__main__":
    main()