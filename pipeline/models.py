"""
pipeline/models.py
LLM Factory supporting Local Ollama (e.g. qwen2.5:3b), Cloud OpenAI/Compatible APIs,
and a Mock/Simulation mode for rapid local demonstrations.
"""

import os
from typing import Optional, List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_available_ollama_models(host: str = "http://localhost:11434") -> List[str]:
    """Retrieve locally installed Ollama models, filtering out embedding models."""
    try:
        import ollama
        client = ollama.Client(host=host)
        models_data = client.list()
        # Handle dict or object response structure
        models_list = models_data.get("models", []) if isinstance(models_data, dict) else getattr(models_data, "models", [])
        
        valid_models = []
        for m in models_list:
            name = m["model"] if isinstance(m, dict) else getattr(m, "model", "")
            # Filter out known embedding models
            if "embed" not in name.lower():
                valid_models.append(name)
        
        if not valid_models:
            valid_models = ["qwen2.5:3b"]
        return valid_models
    except Exception:
        # Fallback default
        return ["qwen2.5:3b", "llama3.2:3b", "mistral:latest"]


def check_ollama_status(host: str = "http://localhost:11434") -> Dict[str, Any]:
    """Check if Ollama server is running and reachable."""
    try:
        import requests
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            return {"online": True, "message": "Ollama service is active and responsive"}
        return {"online": False, "message": f"Status code: {resp.status_code}"}
    except Exception as e:
        return {"online": False, "message": f"Could not connect: {str(e)}"}


def build_llm(
    provider: str = "ollama",
    model_name: str = "qwen2.5:3b",
    temperature: float = 0.7,
    ollama_host: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseChatModel:
    """
    Factory function to instantiate the appropriate ChatModel.
    
    Args:
        provider: 'ollama', 'openai', or 'mock'
        model_name: e.g. 'qwen2.5:3b' or 'gpt-4o-mini'
        temperature: sampling temperature (0.0 to 1.0)
        ollama_host: Ollama server endpoint
        api_key: Cloud API key (falls back to env var if omitted)
        base_url: Cloud API custom endpoint (e.g. OpenRouter, Groq, DeepSeek)
    """
    provider_clean = provider.strip().lower()
    
    if provider_clean == "ollama":
        return ChatOllama(
            model=model_name or "qwen2.5:3b",
            base_url=ollama_host,
            temperature=temperature,
        )
        
    elif provider_clean in ["openai", "cloud"]:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API Key is required. Please supply it in the UI or set the OPENAI_API_KEY environment variable."
            )
        kwargs: Dict[str, Any] = {
            "model": model_name or "gpt-4o-mini",
            "temperature": temperature,
            "api_key": resolved_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
        
    elif provider_clean == "mock":
        from langchain_community.chat_models.fake import FakeListChatModel
        # Return a lightweight mock model for instant offline walkthroughs
        return FakeListChatModel(
            responses=[
                "Mock Pipeline Stage Response: Synthesizing verified data points...",
                "Mock Outline: Structured 4-section architecture ready.",
                "Mock Draft: Comprehensive narrative following approved guidelines.",
                "{\"grammar_and_syntax\": 18, \"clarity_and_coherence\": 19, \"audience_and_tone\": 18, \"outline_adherence\": 19, \"factuality_and_depth\": 17, \"overall_score\": 91, \"strengths\": [\"Clear flow\", \"Solid structure\"], \"areas_for_improvement\": [\"Include more concrete examples\"], \"editorial_critique\": \"Strong foundation with logical cadence.\", \"needs_refinement\": false}",
                "Mock Publish-ready formatted artifact."
            ]
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
