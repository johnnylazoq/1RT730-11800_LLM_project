"""Vector store for storing and retrieving embeddings."""
import json
import numpy as np
from pathlib import Path
from typing import Any


class VectorStore:
    """In-memory vector store using FAISS."""
    
    def __init__(self, embedding_dim: int = 384):
        """Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (e.g., 384 for MiniLM)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")
        
        self.faiss = faiss
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []
        self.id_to_metadata = {}
    
    def add(self, embeddings: np.ndarray, metadata_list: list[dict]) -> list[int]:
        """Add embeddings and metadata to store.
        
        Args:
            embeddings: Array of shape (n, embedding_dim)
            metadata_list: List of metadata dicts (one per embedding)
            
        Returns:
            List of IDs assigned to the embeddings
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim}D embeddings, got {embeddings.shape[1]}D")
        
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata items")
        
        start_id = self.index.ntotal
        embeddings = np.array(embeddings, dtype=np.float32)
        
        self.index.add(embeddings)
        
        ids = []
        for i, meta in enumerate(metadata_list):
            idx = start_id + i
            self.id_to_metadata[idx] = meta
            ids.append(idx)
        
        return ids
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[int, float, dict]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Single embedding of shape (embedding_dim,)
            k: Number of results to return
            
        Returns:
            List of (id, distance, metadata) tuples, sorted by distance
        """
        if self.index.ntotal == 0:
            return []
        
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx >= 0 and idx in self.id_to_metadata:
                results.append((int(idx), float(dist), self.id_to_metadata[idx]))
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        index_file = path / "index.faiss"
        meta_file = path / "metadata.json"
        
        self.faiss.write_index(self.index, str(index_file))
        
        with open(meta_file, 'w') as f:
            json.dump(self.id_to_metadata, f)
    
    def load(self, path: str):
        """Load vector store from disk."""
        path = Path(path)
        
        index_file = path / "index.faiss"
        meta_file = path / "metadata.json"
        
        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Vector store files not found in {path}")
        
        self.index = self.faiss.read_index(str(index_file))
        
        with open(meta_file, 'r') as f:
            meta_dict = json.load(f)
        
        self.id_to_metadata = {int(k): v for k, v in meta_dict.items()}
    
    def __len__(self) -> int:
        """Return number of vectors in store."""
        return self.index.ntotal
