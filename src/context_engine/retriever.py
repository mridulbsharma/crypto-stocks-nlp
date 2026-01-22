"""
Semantic search retriever for CryptoSentinel.

Provides post retrieval with filtering, citation tracking, and evidence scoring
for grounded AI responses based on Reddit post data.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

from .embeddings import EmbeddingEngine
from .evidence import Citation, EvidenceBudget
from ..utils.preprocessing import TextPreprocessor


@dataclass
class SearchResult:
    """
    A single search result with metadata and scoring.
    
    Attributes:
        post_id: Reddit post ID.
        title: Post title.
        text: Post body text.
        url: Full URL to the post.
        subreddit: Source subreddit name.
        datetime: Post datetime.
        score: Reddit score.
        upvote_ratio: Upvote ratio.
        relevance_score: Search relevance score (0-1).
        sentiment_score: VADER sentiment score.
    """
    post_id: str
    title: str
    text: str
    url: str
    subreddit: str
    datetime: datetime
    score: int
    upvote_ratio: float
    relevance_score: float
    sentiment_score: float = 0.0
    
    def to_citation(self) -> Citation:
        """Convert search result to a citation."""
        snippet = self.text if self.text and self.text != 'notexthere' else self.title
        return Citation(
            post_id=self.post_id,
            url=self.url,
            text_snippet=snippet,
            relevance_score=self.relevance_score,
            subreddit=self.subreddit,
            datetime=self.datetime,
            score=self.score
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "post_id": self.post_id,
            "title": self.title,
            "text": self.text[:500] if self.text else "",
            "url": self.url,
            "subreddit": self.subreddit,
            "datetime": self.datetime.isoformat() if self.datetime else None,
            "score": self.score,
            "upvote_ratio": self.upvote_ratio,
            "relevance_score": round(self.relevance_score, 3),
            "sentiment_score": round(self.sentiment_score, 3)
        }


class PostRetriever:
    """
    Semantic search retriever for Reddit posts.
    
    Loads post data from CSV files, builds embeddings for semantic search,
    and provides filtered retrieval with citation tracking.
    """
    
    SUBREDDIT_MAP = {
        0: "CryptoMoonShots",
        1: "wallstreetbets"
    }
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        embedding_method: str = "tfidf"
    ):
        """
        Initialize the post retriever.
        
        Args:
            data_path: Path to the data directory containing CSV files.
            embedding_method: Embedding method ("tfidf" or "transformer").
        """
        self.data_path = Path(data_path) if data_path else self._default_data_path()
        self.preprocessor = TextPreprocessor()
        self.embedding_engine = EmbeddingEngine(method=embedding_method)
        self.evidence_budget = EvidenceBudget()
        
        self._df: Optional[pd.DataFrame] = None
        self._indexed = False
    
    def _default_data_path(self) -> Path:
        """Get default data directory path."""
        return Path(__file__).parent.parent.parent / "data"
    
    def load_data(self, filename: str = "combined_df.csv") -> "PostRetriever":
        """
        Load post data from CSV file.
        
        Args:
            filename: CSV filename in the data directory.
            
        Returns:
            Self for method chaining.
        """
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        self._df = pd.read_csv(filepath)
        
        if 'datetime' in self._df.columns:
            self._df['datetime'] = pd.to_datetime(self._df['datetime'])
        
        if 'text' in self._df.columns:
            self._df['text'] = self._df['text'].fillna('notexthere')
        
        if 'subreddit' in self._df.columns and self._df['subreddit'].dtype in ['int64', 'float64']:
            self._df['subreddit_name'] = self._df['subreddit'].map(self.SUBREDDIT_MAP)
        else:
            self._df['subreddit_name'] = self._df['subreddit']
        
        return self
    
    def build_index(self) -> "PostRetriever":
        """
        Build the search index from loaded data.
        
        Returns:
            Self for method chaining.
        """
        if self._df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        search_texts = []
        for _, row in self._df.iterrows():
            title = str(row.get('title', ''))
            text = str(row.get('text', ''))
            if text == 'notexthere':
                text = ''
            combined = f"{title} {text}".strip()
            search_texts.append(self.preprocessor.preprocess(combined))
        
        self.embedding_engine.fit(search_texts)
        self._indexed = True
        
        return self
    
    def search_posts(
        self,
        query: str,
        subreddit: Optional[str] = None,
        time_range: Optional[tuple[datetime, datetime]] = None,
        min_score: int = 0,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Search posts with optional filters.
        
        Args:
            query: Search query string.
            subreddit: Filter to specific subreddit (None = all).
            time_range: Filter to (start, end) datetime range.
            min_score: Minimum Reddit score filter.
            k: Maximum number of results to return.
            
        Returns:
            List of SearchResult objects sorted by relevance.
        """
        if not self._indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        processed_query = self.preprocessor.preprocess(query)
        top_k_results = self.embedding_engine.similarity(processed_query, top_k=k * 3)
        
        results = []
        for idx, score in top_k_results:
            row = self._df.iloc[idx]
            
            if subreddit:
                row_subreddit = row.get('subreddit_name', row.get('subreddit', ''))
                if str(row_subreddit).lower() != subreddit.lower():
                    continue
            
            if time_range:
                post_dt = row.get('datetime')
                if pd.notna(post_dt):
                    if post_dt < time_range[0] or post_dt > time_range[1]:
                        continue
            
            if row.get('score', 0) < min_score:
                continue
            
            result = SearchResult(
                post_id=str(row.get('id', '')),
                title=str(row.get('title', '')),
                text=str(row.get('text', '')),
                url=str(row.get('url', '')),
                subreddit=str(row.get('subreddit_name', row.get('subreddit', ''))),
                datetime=row.get('datetime') if pd.notna(row.get('datetime')) else None,
                score=int(row.get('score', 0)),
                upvote_ratio=float(row.get('upvote_ratio', 0)),
                relevance_score=score,
                sentiment_score=float(row.get('sentiment_score', 0)) if 'sentiment_score' in row else 0.0
            )
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def search_with_citations(
        self,
        query: str,
        k: int = 10,
        **filters
    ) -> tuple[List[SearchResult], List[Citation]]:
        """
        Search posts and return both results and citations.
        
        Args:
            query: Search query string.
            k: Maximum number of results.
            **filters: Additional filters (subreddit, time_range, min_score).
            
        Returns:
            Tuple of (search_results, citations).
        """
        results = self.search_posts(query, k=k, **filters)
        citations = []
        
        for result in results:
            citation = result.to_citation()
            if self.evidence_budget.add_citation(citation):
                citations.append(citation)
        
        return results, citations
    
    def get_post_by_id(self, post_id: str) -> Optional[SearchResult]:
        """
        Retrieve a specific post by its ID.
        
        Args:
            post_id: Reddit post ID.
            
        Returns:
            SearchResult or None if not found.
        """
        if self._df is None:
            return None
        
        matches = self._df[self._df['id'] == post_id]
        if matches.empty:
            return None
        
        row = matches.iloc[0]
        return SearchResult(
            post_id=str(row.get('id', '')),
            title=str(row.get('title', '')),
            text=str(row.get('text', '')),
            url=str(row.get('url', '')),
            subreddit=str(row.get('subreddit_name', row.get('subreddit', ''))),
            datetime=row.get('datetime') if pd.notna(row.get('datetime')) else None,
            score=int(row.get('score', 0)),
            upvote_ratio=float(row.get('upvote_ratio', 0)),
            relevance_score=1.0,
            sentiment_score=float(row.get('sentiment_score', 0)) if 'sentiment_score' in row else 0.0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary with data statistics.
        """
        if self._df is None:
            return {"loaded": False}
        
        stats = {
            "loaded": True,
            "total_posts": len(self._df),
            "indexed": self._indexed,
            "embedding_dim": self.embedding_engine.embedding_dim,
            "vocabulary_size": self.embedding_engine.vocabulary_size,
        }
        
        if 'subreddit_name' in self._df.columns:
            stats["posts_by_subreddit"] = self._df['subreddit_name'].value_counts().to_dict()
        
        if 'datetime' in self._df.columns:
            valid_dates = self._df['datetime'].dropna()
            if len(valid_dates) > 0:
                stats["date_range"] = {
                    "earliest": str(valid_dates.min()),
                    "latest": str(valid_dates.max())
                }
        
        return stats
