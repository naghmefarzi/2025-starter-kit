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
    run_id = "run_8"
    max_query_iterations = 5
    debug_mode = True

# run_1 with qwen model max query iter=1

# run_2 citations are not correct, they are descriptions- iteration:2, qwen2.5

#run 3 iteration:2, qwen2.5

#run 4 iteration:2, qwen2.5

#run 5 iteration 5 qwen2.5

# run 6 convince false iteration 2 qwen2.5

# run 7 convince false, roasted article is in every process, iteration 5 qwen2.5. for the tasks only the original articlle is used.

# run 8  convince false, roasted article is in every process, iteration 5 qwen2.5. for the tasks only the both articlles are used.