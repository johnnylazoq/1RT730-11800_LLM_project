"""Embedding model wrapper using sentence-transformers."""

import numpy as np


class EmbeddingModel:
    """Wrapper for embedding model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding model.

        Args:
            model_name: Model identifier from HuggingFace.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lazy load model from HuggingFace."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text.

        Args:
            text: Input text.

        Returns:
            Embedding vector (1D numpy array).
        """
        if not self.model:
            self._load_model()
        embeddings = self.model.encode([text], convert_to_numpy=True)
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            Array of embedding vectors (2D numpy array).
        """
        if not self.model:
            self._load_model()
        return self.model.encode(texts, convert_to_numpy=True)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension.

        Returns:
            Dimension of embeddings.
        """
        if not self.model:
            self._load_model()
        return self.model.get_sentence_embedding_dimension()
