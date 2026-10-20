
import os
from langchain_ollama import ChatOllama


def get_llm(model_name=None):
    return ChatOllama(
        model=model_name or os.getenv("OLLAMA_MODEL", "frob/qwen3.5-instruct:9b"),
        temperature=0,
        client_kwargs={"timeout": float(os.getenv("OLLAMA_TIMEOUT", "120"))},
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        ),
    )
