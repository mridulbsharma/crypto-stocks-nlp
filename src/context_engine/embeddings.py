"""
Embedding engine for semantic search in CryptoSentinel.

Provides text embeddings using TF-IDF or sentence transformers for
similarity-based retrieval of Reddit posts.
"""

from typing import List, Optional, Union
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingEngine:
    """
    Text embedding engine supporting TF-IDF and optional transformer-based embeddings.
    
    Provides functionality for creating embeddings and computing similarity scores
    for semantic search over Reddit posts.
    """
    
    def __init__(
        self,
        method: str = "tfidf",
        model_name: str = "all-MiniLM-L6-v2",
        max_features: int = 10000,
        ngram_range: tuple = (1, 2)
    ):
        """
        Initialize the embedding engine.
        
        Args:
            method: Embedding method - "tfidf" or "transformer".
            model_name: Sentence transformer model name (if method="transformer").
            max_features: Maximum vocabulary size for TF-IDF.
            ngram_range: N-gram range for TF-IDF vectorization.
        """
        self.method = method
        self.model_name = model_name
        self._embeddings: Optional[NDArray] = None
        self._texts: List[str] = []
        
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                max_df=0.5,
                min_df=2
            )
            self.model = None
        elif method == "transformer":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.vectorizer = None
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for transformer embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        else:
            raise ValueError(f"Unknown embedding method: {method}")
    
    def fit(self, texts: List[str]) -> "EmbeddingEngine":
        """
        Fit the embedding engine on a corpus of texts.
        
        Args:
            texts: List of text documents to fit on.
            
        Returns:
            Self for method chaining.
        """
        self._texts = texts
        
        if self.method == "tfidf":
            self._embeddings = self.vectorizer.fit_transform(texts).toarray()
        else:
            self._embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        return self
    
    def embed(self, texts: Union[str, List[str]]) -> NDArray:
        """
        Generate embeddings for new texts.
        
        Args:
            texts: Single text or list of texts to embed.
            
        Returns:
            Numpy array of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.method == "tfidf":
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted. Call fit() first.")
            return self.vectorizer.transform(texts).toarray()
        else:
            return self.model.encode(texts, convert_to_numpy=True)
    
    def similarity(
        self,
        query: str,
        top_k: int = 10
    ) -> List[tuple[int, float]]:
        """
        Find most similar documents to a query.
        
        Args:
            query: Query text string.
            top_k: Number of top results to return.
            
        Returns:
            List of (index, similarity_score) tuples, sorted by score descending.
        """
        if self._embeddings is None:
            raise ValueError("No embeddings available. Call fit() first.")
        
        query_embedding = self.embed(query)
        similarities = cosine_similarity(query_embedding, self._embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def batch_similarity(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[tuple[int, float]]]:
        """
        Find most similar documents for multiple queries.
        
        Args:
            queries: List of query texts.
            top_k: Number of top results per query.
            
        Returns:
            List of results for each query.
        """
        return [self.similarity(q, top_k) for q in queries]
    
    @property
    def vocabulary_size(self) -> int:
        """Return the vocabulary size (TF-IDF only)."""
        if self.method == "tfidf" and self.vectorizer:
            vocab = getattr(self.vectorizer, 'vocabulary_', None)
            return len(vocab) if vocab else 0
        return 0
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        if self._embeddings is not None:
            return self._embeddings.shape[1]
        return 0
    
    def save(self, path: Path) -> None:
        """
        Save embeddings and model state to disk.
        
        Args:
            path: Directory path to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self._embeddings is not None:
            np.save(path / "embeddings.npy", self._embeddings)
        
        if self.method == "tfidf" and self.vectorizer:
            import pickle
            with open(path / "vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
    
    def load(self, path: Path) -> "EmbeddingEngine":
        """
        Load embeddings and model state from disk.
        
        Args:
            path: Directory path to load from.
            
        Returns:
            Self for method chaining.
        """
        path = Path(path)
        
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            self._embeddings = np.load(embeddings_path)
        
        if self.method == "tfidf":
            vectorizer_path = path / "vectorizer.pkl"
            if vectorizer_path.exists():
                import pickle
                with open(vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
        
        return self
