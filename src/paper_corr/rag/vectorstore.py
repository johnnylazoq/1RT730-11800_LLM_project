"""Build and query vector index using FAISS."""

import pickle
from pathlib import Path

import numpy as np

from paper_corr.rag.chunker import Chunk


class VectorStore:
    """Vector store for paper chunks using FAISS."""

    def __init__(self, embedding_dim: int = 384):
        """Initialize vector store.

        Args:
            embedding_dim: Dimension of embeddings.
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunk_store = {}  # chunk_id -> Chunk
        self._init_index()

    def _init_index(self):
        """Initialize FAISS index."""
        try:
            import faiss

            self.index = faiss.IndexFlatL2(self.embedding_dim)
        except ImportError:
            raise ImportError("faiss-cpu not installed. Install with: pip install faiss-cpu")

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray):
        """Add chunks to index.

        Args:
            chunks: List of Chunk objects.
            embeddings: Embedding vectors (N x embedding_dim).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) mismatch")

        # Normalize embeddings to unit vectors for cosine similarity
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))

        # Store chunks
        for i, chunk in enumerate(chunks):
            self.chunk_store[chunk.chunk_id] = chunk

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector.
            k: Number of results.

        Returns:
            List of (Chunk, similarity_score) tuples, ranked by similarity.
        """
        # Normalize query embedding
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search FAISS index (L2 distance)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # No result
                continue

            # Convert L2 distance to cosine similarity
            # L2 distance = 2 * (1 - cosine_similarity) for unit vectors
            similarity = 1 - (distances[0][i] / 2)
            similarity = max(0, min(1, similarity))  # Clamp to [0, 1]

            # Get chunk from store
            chunk_list = list(self.chunk_store.values())
            if idx < len(chunk_list):
                chunk = chunk_list[idx]
                results.append((chunk, similarity))

        return results

    def save(self, index_path: str | Path):
        """Save index to disk.

        Args:
            index_path: Directory to save index.
        """
        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        try:
            import faiss

            # Save FAISS index
            faiss.write_index(self.index, str(index_path / "index.faiss"))

            # Save chunks metadata
            with open(index_path / "chunks.pkl", "wb") as f:
                pickle.dump(self.chunk_store, f)
        except ImportError:
            raise ImportError("faiss-cpu not installed")

    def load(self, index_path: str | Path):
        """Load index from disk.

        Args:
            index_path: Directory to load index from.
        """
        index_path = Path(index_path)

        try:
            import faiss

            # Load FAISS index
            self.index = faiss.read_index(str(index_path / "index.faiss"))

            # Load chunks metadata
            with open(index_path / "chunks.pkl", "rb") as f:
                self.chunk_store = pickle.load(f)
        except ImportError:
            raise ImportError("faiss-cpu not installed")
        except FileNotFoundError:
            raise FileNotFoundError(f"Index files not found in {index_path}")
