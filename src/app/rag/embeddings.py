"""Generate embeddings for text chunks."""
import numpy as np
from typing import Any


class EmbeddingModel:
    """Wrapper around sentence-transformers for generating embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model.
        
        Args:
            model_name: Hugging Face model identifier
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return np.array(embeddings, dtype=np.float32)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score between -1 and 1 (typically 0 to 1)
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
