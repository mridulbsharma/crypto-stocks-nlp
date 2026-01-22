"""
MCP Tools for CryptoSentinel.

Defines the tool interfaces for Reddit search, sentiment aggregation,
topic clustering, classification, and brief generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..context_engine.retriever import PostRetriever, SearchResult
from ..context_engine.evidence import Citation
from ..models.classifier import EnsembleClassifier, ClassificationResult
from ..reasoning.socratic_refine import SocraticRefiner


@dataclass
class Post:
    """A Reddit post returned from search."""
    id: str
    title: str
    text: str
    url: str
    subreddit: str
    datetime: Optional[str]
    score: int
    relevance: float


@dataclass
class SentimentReport:
    """Aggregated sentiment report for a topic."""
    topic: str
    subreddit: Optional[str]
    total_posts: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_sentiment: float
    sentiment_label: str
    top_positive: List[Post]
    top_negative: List[Post]


@dataclass
class TopicCluster:
    """A cluster of related topics."""
    cluster_id: int
    label: str
    keywords: List[str]
    post_count: int
    avg_sentiment: float
    representative_posts: List[Post]


@dataclass
class NarrativeBrief:
    """A grounded narrative brief with citations."""
    topic: str
    summary: str
    key_insights: List[str]
    sentiment_overview: str
    citations: List[Citation]
    confidence: float
    groundedness: str
    warnings: List[str]


class CryptoSentinelTools:
    """
    Collection of MCP-compatible tools for CryptoSentinel.
    
    Provides high-level interfaces for Reddit analysis operations.
    """
    
    def __init__(self, retriever: Optional[PostRetriever] = None):
        """
        Initialize the tools.
        
        Args:
            retriever: PostRetriever instance (creates default if None).
        """
        self._retriever = retriever
        self._classifier: Optional[EnsembleClassifier] = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Ensure retriever is loaded and indexed."""
        if self._initialized:
            return
        
        if self._retriever is None:
            self._retriever = PostRetriever()
        
        try:
            self._retriever.load_data()
            self._retriever.build_index()
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tools: {e}")
    
    def _search_result_to_post(self, result: SearchResult) -> Post:
        """Convert SearchResult to Post."""
        return Post(
            id=result.post_id,
            title=result.title,
            text=result.text[:500] if result.text else "",
            url=result.url,
            subreddit=result.subreddit,
            datetime=result.datetime.isoformat() if result.datetime else None,
            score=result.score,
            relevance=result.relevance_score
        )


def reddit_search(
    query: str,
    subreddit: Optional[str] = None,
    time_range: Optional[str] = None,
    min_score: int = 0,
    max_results: int = 10,
    tools: Optional[CryptoSentinelTools] = None
) -> List[Post]:
    """
    Search Reddit posts with optional filters.
    
    Args:
        query: Search query string.
        subreddit: Filter to specific subreddit ("wallstreetbets" or "CryptoMoonShots").
        time_range: Time range filter ("day", "week", "month", "year").
        min_score: Minimum Reddit score filter.
        max_results: Maximum number of results.
        tools: CryptoSentinelTools instance.
        
    Returns:
        List of matching Post objects.
    """
    if tools is None:
        tools = CryptoSentinelTools()
    
    tools._ensure_initialized()
    
    time_filter = None
    if time_range:
        now = datetime.now()
        days_map = {"day": 1, "week": 7, "month": 30, "year": 365}
        if time_range in days_map:
            from datetime import timedelta
            start = now - timedelta(days=days_map[time_range])
            time_filter = (start, now)
    
    results = tools._retriever.search_posts(
        query=query,
        subreddit=subreddit,
        time_range=time_filter,
        min_score=min_score,
        k=max_results
    )
    
    return [tools._search_result_to_post(r) for r in results]


def sentiment_aggregate(
    topic: str,
    subreddit: Optional[str] = None,
    tools: Optional[CryptoSentinelTools] = None
) -> SentimentReport:
    """
    Aggregate sentiment for a topic across posts.
    
    Args:
        topic: Topic to analyze sentiment for.
        subreddit: Filter to specific subreddit.
        tools: CryptoSentinelTools instance.
        
    Returns:
        SentimentReport with aggregated sentiment analysis.
    """
    if tools is None:
        tools = CryptoSentinelTools()
    
    tools._ensure_initialized()
    
    results = tools._retriever.search_posts(
        query=topic,
        subreddit=subreddit,
        k=50
    )
    
    if not results:
        return SentimentReport(
            topic=topic,
            subreddit=subreddit,
            total_posts=0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            average_sentiment=0.0,
            sentiment_label="unknown",
            top_positive=[],
            top_negative=[]
        )
    
    positive = [r for r in results if r.sentiment_score > 0.05]
    negative = [r for r in results if r.sentiment_score < -0.05]
    neutral = [r for r in results if -0.05 <= r.sentiment_score <= 0.05]
    
    avg_sentiment = sum(r.sentiment_score for r in results) / len(results)
    
    if avg_sentiment > 0.1:
        label = "bullish"
    elif avg_sentiment < -0.1:
        label = "bearish"
    else:
        label = "neutral"
    
    positive_sorted = sorted(positive, key=lambda x: x.sentiment_score, reverse=True)
    negative_sorted = sorted(negative, key=lambda x: x.sentiment_score)
    
    return SentimentReport(
        topic=topic,
        subreddit=subreddit,
        total_posts=len(results),
        positive_count=len(positive),
        negative_count=len(negative),
        neutral_count=len(neutral),
        average_sentiment=avg_sentiment,
        sentiment_label=label,
        top_positive=[tools._search_result_to_post(r) for r in positive_sorted[:3]],
        top_negative=[tools._search_result_to_post(r) for r in negative_sorted[:3]]
    )


def topic_clusters(
    subreddit: Optional[str] = None,
    n_clusters: int = 5,
    tools: Optional[CryptoSentinelTools] = None
) -> List[TopicCluster]:
    """
    Extract main topic clusters from posts.
    
    Args:
        subreddit: Filter to specific subreddit.
        n_clusters: Number of clusters to extract.
        tools: CryptoSentinelTools instance.
        
    Returns:
        List of TopicCluster objects.
    """
    if tools is None:
        tools = CryptoSentinelTools()
    
    tools._ensure_initialized()
    
    results = tools._retriever.search_posts(
        query="",
        subreddit=subreddit,
        k=100
    )
    
    if not results:
        return []
    
    keyword_groups = {
        0: {"label": "Meme Stocks", "keywords": ["gme", "amc", "yolo", "diamond", "hands", "ape"]},
        1: {"label": "Bitcoin & Major Crypto", "keywords": ["btc", "bitcoin", "eth", "ethereum", "crypto"]},
        2: {"label": "Options Trading", "keywords": ["calls", "puts", "options", "expiry", "strike"]},
        3: {"label": "Altcoins & DeFi", "keywords": ["defi", "altcoin", "token", "moon", "gem"]},
        4: {"label": "Market Analysis", "keywords": ["market", "bull", "bear", "trend", "analysis"]}
    }
    
    clusters = []
    for i in range(min(n_clusters, len(keyword_groups))):
        group = keyword_groups[i]
        
        matching = [r for r in results if any(
            kw in r.title.lower() or kw in r.text.lower() 
            for kw in group["keywords"]
        )]
        
        if matching:
            avg_sent = sum(r.sentiment_score for r in matching) / len(matching)
        else:
            avg_sent = 0.0
        
        clusters.append(TopicCluster(
            cluster_id=i,
            label=group["label"],
            keywords=group["keywords"],
            post_count=len(matching),
            avg_sentiment=avg_sent,
            representative_posts=[tools._search_result_to_post(r) for r in matching[:3]]
        ))
    
    clusters.sort(key=lambda x: x.post_count, reverse=True)
    return clusters


def classify_post(
    title: str,
    text: Optional[str] = None,
    tools: Optional[CryptoSentinelTools] = None
) -> ClassificationResult:
    """
    Classify if a post is crypto or stocks related.
    
    Args:
        title: Post title.
        text: Post body text (optional).
        tools: CryptoSentinelTools instance.
        
    Returns:
        ClassificationResult with prediction and explanation.
    """
    if tools is None:
        tools = CryptoSentinelTools()
    
    if tools._classifier is None:
        tools._classifier = EnsembleClassifier()
        tools._ensure_initialized()
        
        if tools._retriever._df is not None:
            df = tools._retriever._df
            texts = df['title'].tolist()
            labels = df['subreddit'].tolist() if 'subreddit' in df.columns else [0] * len(texts)
            
            if isinstance(labels[0], str):
                labels = [1 if 'wallstreetbets' in str(l).lower() else 0 for l in labels]
            
            tools._classifier.fit(texts, labels, preprocess=True)
    
    full_text = f"{title} {text}" if text else title
    return tools._classifier.predict(full_text)


def generate_brief(
    topic: str,
    max_sources: int = 5,
    tools: Optional[CryptoSentinelTools] = None
) -> NarrativeBrief:
    """
    Generate a grounded narrative brief with citations.
    
    Args:
        topic: Topic to generate brief about.
        max_sources: Maximum number of sources to include.
        tools: CryptoSentinelTools instance.
        
    Returns:
        NarrativeBrief with grounded insights and citations.
    """
    if tools is None:
        tools = CryptoSentinelTools()
    
    tools._ensure_initialized()
    
    refiner = SocraticRefiner(tools._retriever)
    refined = refiner.refine(topic)
    
    if refined.refused:
        return NarrativeBrief(
            topic=topic,
            summary=f"Unable to generate a grounded brief: {refined.refusal_reason}",
            key_insights=[],
            sentiment_overview="Unknown due to insufficient evidence",
            citations=[],
            confidence=0.0,
            groundedness="insufficient",
            warnings=["Response refused due to insufficient grounding"]
        )
    
    insights = []
    if refined.verification.claims_verified > 0:
        insights.append(f"Analysis based on {len(refined.citations)} verified sources")
    
    for vr in refined.verification.verification_results:
        if vr.is_valid and vr.reasoning:
            insights.append(vr.reasoning)
    
    sentiment_scores = [c.score for c in refined.citations if hasattr(c, 'score')]
    if sentiment_scores:
        avg = sum(sentiment_scores) / len(sentiment_scores)
        if avg > 10:
            sentiment = "Generally positive community reception"
        elif avg < 0:
            sentiment = "Mixed or negative community reception"
        else:
            sentiment = "Neutral community reception"
    else:
        sentiment = "Sentiment data unavailable"
    
    warnings = []
    if refined.groundedness_score < 0.7:
        warnings.append("Some claims may be speculative")
    if len(refined.citations) < 3:
        warnings.append("Limited source diversity")
    
    groundedness = "high" if refined.groundedness_score > 0.8 else \
                   "medium" if refined.groundedness_score > 0.5 else "low"
    
    return NarrativeBrief(
        topic=topic,
        summary=refined.answer,
        key_insights=insights[:5],
        sentiment_overview=sentiment,
        citations=refined.citations[:max_sources],
        confidence=refined.confidence,
        groundedness=groundedness,
        warnings=warnings
    )
