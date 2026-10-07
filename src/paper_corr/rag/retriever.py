"""Retrieve papers and chunks from vector store."""

from paper_corr.rag.chunker import Chunk
from paper_corr.rag.embeddings import EmbeddingModel
from paper_corr.rag.loader import Paper
from paper_corr.rag.vectorstore import VectorStore


class Retriever:
    """Retrieve papers and chunks using vector similarity."""

    def __init__(self, vectorstore: VectorStore, embedding_model: EmbeddingModel, papers: dict[str, Paper]):
        """Initialize retriever.

        Args:
            vectorstore: VectorStore instance.
            embedding_model: EmbeddingModel instance.
            papers: Dictionary of paper_id -> Paper objects.
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.papers = papers

    def retrieve_papers(self, query: str, k: int = 5) -> list[tuple[Paper, float]]:
        """Retrieve papers matching query.

        Args:
            query: Query string.
            k: Number of papers to return.

        Returns:
            List of (Paper, average_similarity) tuples, ranked by relevance.
        """
        # Embed query
        query_embedding = self.embedding_model.embed_text(query)

        # Search for similar chunks
        chunk_results = self.vectorstore.search(query_embedding, k=k * 3)

        # Group by paper and average similarity
        paper_scores = {}
        for chunk, similarity in chunk_results:
            if chunk.paper_id not in paper_scores:
                paper_scores[chunk.paper_id] = []
            paper_scores[chunk.paper_id].append(similarity)

        # Average scores and get papers
        results = []
        for paper_id, scores in sorted(
            paper_scores.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True,
        )[:k]:
            if paper_id in self.papers:
                avg_score = sum(scores) / len(scores)
                results.append((self.papers[paper_id], avg_score))

        return results

    def retrieve_chunks(self, paper_id: str, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve chunks from a specific paper matching query.

        Args:
            paper_id: Paper ID.
            query: Query string.
            k: Number of chunks to return.

        Returns:
            List of Chunk objects, ranked by relevance.
        """
        # Embed query
        query_embedding = self.embedding_model.embed_text(query)

        # Search for similar chunks
        chunk_results = self.vectorstore.search(query_embedding, k=k * 2)

        # Filter by paper_id
        results = [chunk for chunk, _ in chunk_results if chunk.paper_id == paper_id][:k]

        return results
