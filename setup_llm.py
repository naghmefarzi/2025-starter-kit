#!/usr/bin/env python3
"""
Simple setup script for Ollama with Llama models.
"""
import os
import sys
import subprocess
import requests
import time


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install required Python dependencies."""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)


def setup_ollama():
    """Setup Ollama with Llama models."""
    print("\n=== Setting up Ollama ===")
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama is already installed")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Ollama not found. Please install Ollama first:")
        print("Visit: https://ollama.ai/download")
        print("Or run: curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    
    # Check if Ollama service is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama service is running")
        else:
            print("Ollama service is not responding properly")
            return False
    except requests.RequestException:
        print("Starting Ollama service...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # Wait for service to start
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                print("✓ Ollama service started successfully")
            else:
                print("Failed to start Ollama service")
                return False
        except Exception as e:
            print(f"Error starting Ollama: {e}")
            return False
    
    # Check available models
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json().get('models', [])
        model_names = [model['name'] for model in models]
        
        llama_models = [
            # "llama3.1:70b-instruct",
            "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M"
        ]
        
        available_llama = [m for m in llama_models if m in model_names]
        
        if available_llama:
            print(f"✓ Found Llama models: {available_llama}")
        else:
            print("No Llama models found. Available models:")
            for model in model_names:
                print(f"  - {model}")
            
            print("\nRecommended models:")
            print("  - hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M (faster, less memory)")
            print("  - llama3.1:70b-instruct (better quality, more memory)")
            
            print("\nTo pull a model, run:")
            print("  ollama pull hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M")
            print("  ollama pull llama3.1:70b-instruct")
    
    except Exception as e:
        print(f"Error checking models: {e}")
        return False
    
    return True


def create_env_file(model_name):
    """Create .env file with model configuration."""
    env_content = f"""# Model Configuration
MODEL_NAME={model_name}
DEBUG_MODE=false
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"✓ Created .env file with model: {model_name}")


def main():
    """Main setup function."""
    print("TREC DRAGUN System - Simple Ollama Setup")
    print("=" * 40)
    
    check_python_version()
    install_dependencies()
    
    if setup_ollama():
        print("\nChoose your model:")
        print("1. hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M (faster, 8GB RAM)")
        # print("2. llama3.1:70b-instruct (better quality, 40GB+ RAM)")
        
        choice = input("\nSelect model (1-2): ").strip()
        
        if choice == "1":
            model = "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M"
        elif choice == "2":
            model = "llama3.1:70b-instruct"
        else:
            print("Invalid choice, defaulting to 8B model")
            model = "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M"
        
        create_env_file(model)
        print(f"\n✓ Setup complete!")
        print(f"Using model: {model}")
        print("\nTo test your setup: python test_llm.py")
        print("To run the system: python main.py")
    else:
        print("\n❌ Setup failed")


if __name__ == "__main__":
    main()