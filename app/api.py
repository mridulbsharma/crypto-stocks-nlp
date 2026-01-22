"""
FastAPI endpoints for CryptoSentinel.

Provides REST API endpoints for all CryptoSentinel functionality,
enabling integration with external applications and services.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="CryptoSentinel API",
    description="Agentic AI for Grounded Crypto & Stock Market Intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_tools = None


def get_tools():
    """Get or initialize CryptoSentinel tools."""
    global _tools
    if _tools is None:
        from src.mcp_server.tools import CryptoSentinelTools
        _tools = CryptoSentinelTools()
        _tools._ensure_initialized()
    return _tools


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query string")
    subreddit: Optional[str] = Field(None, description="Filter to subreddit")
    time_range: Optional[str] = Field(None, description="Time range: day, week, month, year")
    min_score: int = Field(0, description="Minimum Reddit score")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results")


class SentimentRequest(BaseModel):
    """Request model for sentiment endpoint."""
    topic: str = Field(..., description="Topic to analyze")
    subreddit: Optional[str] = Field(None, description="Filter to subreddit")


class ClassifyRequest(BaseModel):
    """Request model for classification endpoint."""
    title: str = Field(..., description="Post title")
    text: Optional[str] = Field(None, description="Post body text")


class BriefRequest(BaseModel):
    """Request model for brief generation endpoint."""
    topic: str = Field(..., description="Topic for brief")
    max_sources: int = Field(5, ge=1, le=20, description="Maximum sources")


class ReasoningRequest(BaseModel):
    """Request model for full reasoning endpoint."""
    question: str = Field(..., description="Question to answer")


class PostResponse(BaseModel):
    """Response model for a post."""
    id: str
    title: str
    text: str
    url: str
    subreddit: str
    datetime: Optional[str]
    score: int
    relevance: float


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    topic: str
    subreddit: Optional[str]
    total_posts: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_sentiment: float
    sentiment_label: str
    top_positive: List[PostResponse]
    top_negative: List[PostResponse]


class ClassificationResponse(BaseModel):
    """Response model for classification."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    top_features: List[Dict[str, Any]]


class BriefResponse(BaseModel):
    """Response model for brief generation."""
    topic: str
    summary: str
    key_insights: List[str]
    sentiment_overview: str
    citations: List[Dict[str, Any]]
    confidence: float
    groundedness: str
    warnings: List[str]


class ReasoningResponse(BaseModel):
    """Response model for full reasoning."""
    question: str
    answer: str
    confidence: float
    groundedness_score: float
    citations: List[Dict[str, Any]]
    verification: Dict[str, Any]
    trace: List[Dict[str, Any]]
    refused: bool
    refusal_reason: Optional[str]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    data_loaded: bool
    total_posts: int


@app.get("/", tags=["Info"])
async def root():
    """API root endpoint."""
    return {
        "name": "CryptoSentinel API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint."""
    try:
        tools = get_tools()
        stats = tools._retriever.get_statistics()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            data_loaded=stats.get("loaded", False),
            total_posts=stats.get("total_posts", 0)
        )
    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)}",
            version="1.0.0",
            data_loaded=False,
            total_posts=0
        )


@app.post("/search", response_model=List[PostResponse], tags=["Analysis"])
async def search_posts(request: SearchRequest):
    """
    Search Reddit posts with filters.
    
    Returns posts from r/wallstreetbets and r/CryptoMoonShots
    matching the query with optional filters.
    """
    try:
        from src.mcp_server.tools import reddit_search
        
        tools = get_tools()
        posts = reddit_search(
            query=request.query,
            subreddit=request.subreddit,
            time_range=request.time_range,
            min_score=request.min_score,
            max_results=request.max_results,
            tools=tools
        )
        
        return [
            PostResponse(
                id=p.id,
                title=p.title,
                text=p.text,
                url=p.url,
                subreddit=p.subreddit,
                datetime=p.datetime,
                score=p.score,
                relevance=round(p.relevance, 3)
            )
            for p in posts
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment", response_model=SentimentResponse, tags=["Analysis"])
async def analyze_sentiment(request: SentimentRequest):
    """
    Aggregate sentiment for a topic.
    
    Analyzes posts related to the topic and returns
    sentiment distribution with top positive/negative examples.
    """
    try:
        from src.mcp_server.tools import sentiment_aggregate
        
        tools = get_tools()
        report = sentiment_aggregate(
            topic=request.topic,
            subreddit=request.subreddit,
            tools=tools
        )
        
        return SentimentResponse(
            topic=report.topic,
            subreddit=report.subreddit,
            total_posts=report.total_posts,
            positive_count=report.positive_count,
            negative_count=report.negative_count,
            neutral_count=report.neutral_count,
            average_sentiment=round(report.average_sentiment, 3),
            sentiment_label=report.sentiment_label,
            top_positive=[
                PostResponse(
                    id=p.id, title=p.title, text=p.text, url=p.url,
                    subreddit=p.subreddit, datetime=p.datetime,
                    score=p.score, relevance=round(p.relevance, 3)
                )
                for p in report.top_positive
            ],
            top_negative=[
                PostResponse(
                    id=p.id, title=p.title, text=p.text, url=p.url,
                    subreddit=p.subreddit, datetime=p.datetime,
                    score=p.score, relevance=round(p.relevance, 3)
                )
                for p in report.top_negative
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify", response_model=ClassificationResponse, tags=["Analysis"])
async def classify_post(request: ClassifyRequest):
    """
    Classify a post as crypto or stocks.
    
    Uses the ensemble NB + LR classifier to determine
    if content is cryptocurrency or stock market related.
    """
    try:
        from src.mcp_server.tools import classify_post as classify
        
        tools = get_tools()
        result = classify(
            title=request.title,
            text=request.text,
            tools=tools
        )
        
        return ClassificationResponse(
            predicted_class=result.predicted_class,
            confidence=round(result.confidence, 3),
            probabilities={k: round(v, 3) for k, v in result.probabilities.items()},
            top_features=[
                {"feature": e.feature, "weight": round(e.weight, 4), "direction": e.direction}
                for e in result.explanations[:5]
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/brief", response_model=BriefResponse, tags=["Analysis"])
async def generate_brief(request: BriefRequest):
    """
    Generate a grounded narrative brief.
    
    Uses the Socratic refinement loop to produce
    a grounded brief with citations and confidence scores.
    """
    try:
        from src.mcp_server.tools import generate_brief as gen_brief
        
        tools = get_tools()
        brief = gen_brief(
            topic=request.topic,
            max_sources=request.max_sources,
            tools=tools
        )
        
        return BriefResponse(
            topic=brief.topic,
            summary=brief.summary,
            key_insights=brief.key_insights,
            sentiment_overview=brief.sentiment_overview,
            citations=[c.to_dict() for c in brief.citations],
            confidence=round(brief.confidence, 3),
            groundedness=brief.groundedness,
            warnings=brief.warnings
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reason", response_model=ReasoningResponse, tags=["Reasoning"])
async def full_reasoning(request: ReasoningRequest):
    """
    Execute full Socratic reasoning.
    
    Runs the complete Plan → Execute → Verify → Finalize
    loop with full reasoning trace.
    """
    try:
        from src.reasoning.socratic_refine import SocraticRefiner
        
        tools = get_tools()
        refiner = SocraticRefiner(tools._retriever)
        result = refiner.refine(request.question)
        
        return ReasoningResponse(
            question=result.question,
            answer=result.answer,
            confidence=round(result.confidence, 3),
            groundedness_score=round(result.groundedness_score, 3),
            citations=[c.to_dict() for c in result.citations],
            verification={
                "claims_verified": result.verification.claims_verified,
                "claims_rejected": result.verification.claims_rejected,
                "gaps_found": result.verification.gaps_found,
                "contradictions": result.verification.contradictions,
                "overall_confidence": round(result.verification.overall_confidence, 3)
            },
            trace=result.trace,
            refused=result.refused,
            refusal_reason=result.refusal_reason
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clusters", tags=["Analysis"])
async def get_topic_clusters(
    subreddit: Optional[str] = Query(None, description="Filter to subreddit"),
    n_clusters: int = Query(5, ge=1, le=10, description="Number of clusters")
):
    """
    Extract topic clusters from posts.
    
    Identifies main discussion themes in the subreddit(s).
    """
    try:
        from src.mcp_server.tools import topic_clusters
        
        tools = get_tools()
        clusters = topic_clusters(
            subreddit=subreddit,
            n_clusters=n_clusters,
            tools=tools
        )
        
        return [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "keywords": c.keywords,
                "post_count": c.post_count,
                "avg_sentiment": round(c.avg_sentiment, 3),
                "representative_posts": [
                    {"id": p.id, "title": p.title, "url": p.url}
                    for p in c.representative_posts
                ]
            }
            for c in clusters
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools", tags=["MCP"])
async def list_tools():
    """
    List available MCP tools.
    
    Returns tool definitions compatible with MCP protocol.
    """
    from src.mcp_server.server import CryptoSentinelMCPServer
    from dataclasses import asdict
    
    server = CryptoSentinelMCPServer()
    tools = server.get_tool_definitions()
    return {"tools": [asdict(t) for t in tools]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
