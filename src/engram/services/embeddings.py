"""Embedding generation for entity resolution.

Uses OpenAI's text-embedding-3-small model (1536 dimensions, cosine similarity).
Supports async embedding generation for integration with extraction pipeline.
"""

from __future__ import annotations

from openai import AsyncOpenAI


class EmbeddingService:
    """Generate embeddings for entity resolution via vector similarity.

    Integrates with Neo4j vector index for duplicate detection.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        """Initialize embedding service.

        Args:
            model: OpenAI embedding model (default: text-embedding-3-small, 1536 dims)
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed (e.g., entity canonical name + aliases)

        Returns:
            List of 1536 floats (for text-embedding-3-small)
        """
        response = await self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding
