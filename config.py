"""
Simple configuration for vLLM with Llama models.
"""
import os


class CONFIG:
    # LLM Configuration - Simple vLLM setup
    # model_name = os.getenv('MODEL_NAME', 'hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M')

    model_name = os.getenv('MODEL_NAME', 'qwen2.5:7b')
    base_url = "http://localhost:11434"
    temperature = 0.3
    max_tokens = 4096

    # System Configuration
    team_id = "TREMA_UNH"
    run_id = "run_1"
    max_query_iterations = 5
    debug_mode = True

