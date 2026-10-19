
import os
from langchain_ollama import ChatOllama


def get_llm():
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "frob/qwen3.5-instruct:9b"),
        temperature=0,
        num_ctx=num_ctx,
        num_predict=min(2048, num_ctx // 4),
        client_kwargs={"timeout": float(os.getenv("OLLAMA_TIMEOUT", "120"))},
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        ),
    )
