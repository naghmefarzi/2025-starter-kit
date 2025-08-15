"""
Simple configuration for vLLM with Llama models.
"""
import os


class CONFIG:
    # LLM Configuration - Simple vLLM setup
    # model_name = os.getenv('MODEL_NAME', 'hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M')

    model_name = os.getenv('MODEL_NAME', 'qwen2.5:7b')
    
    # model_name = os.getenv("MODEL_NAME", 'gpt-oss:20b')
    # model_name = os.getenv("MODEL_NAME", "qwen:14b")
    base_url = "http://localhost:11434"
    temperature = 0.3
    max_tokens = 100000

    # System Configuration
    team_id = "TREMA_UNH"
    run_id = "run_2"
    max_query_iterations = 1
    debug_mode = True

# run_1 with qwen model max query iter=1

