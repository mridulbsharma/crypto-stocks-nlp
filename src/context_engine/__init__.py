"""Context Engine - Semantic search and evidence management for CryptoSentinel."""

from .retriever import PostRetriever, SearchResult
from .embeddings import EmbeddingEngine
from .evidence import EvidenceBudget, Citation, ClaimVerificationResult

__all__ = [
    "PostRetriever",
    "SearchResult",
    "EmbeddingEngine",
    "EvidenceBudget",
    "Citation",
    "ClaimVerificationResult",
]
