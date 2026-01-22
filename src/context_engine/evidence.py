"""
Evidence management for grounded AI responses in CryptoSentinel.

Provides citation tracking, evidence budget management, and claim validation
to ensure AI-generated insights are properly grounded in source data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


class GroundednessLevel(Enum):
    """Classification of claim groundedness."""
    GROUNDED = "grounded"
    SPECULATION = "speculation"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class Citation:
    """
    A citation linking a claim to its source evidence.
    
    Attributes:
        post_id: Reddit post ID.
        url: Full URL to the Reddit post.
        text_snippet: Relevant excerpt from the post.
        relevance_score: Similarity/relevance score (0-1).
        subreddit: Source subreddit name.
        datetime: Original post datetime.
        score: Reddit score (upvotes - downvotes).
    """
    post_id: str
    url: str
    text_snippet: str
    relevance_score: float
    subreddit: str
    datetime: Optional[datetime] = None
    score: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary representation."""
        return {
            "post_id": self.post_id,
            "url": self.url,
            "text_snippet": self.text_snippet[:200] + "..." if len(self.text_snippet) > 200 else self.text_snippet,
            "relevance_score": round(self.relevance_score, 3),
            "subreddit": self.subreddit,
            "datetime": self.datetime.isoformat() if self.datetime else None,
            "score": self.score
        }
    
    def format_markdown(self) -> str:
        """Format citation as markdown reference."""
        date_str = self.datetime.strftime("%Y-%m-%d") if self.datetime else "Unknown"
        return f"[{self.subreddit} - {date_str}]({self.url})"


@dataclass
class ClaimVerificationResult:
    """
    Result of verifying a claim against evidence.
    
    Attributes:
        claim: The original claim text.
        groundedness: Level of groundedness.
        citations: Supporting citations.
        confidence: Confidence in the verification (0-1).
        reasoning: Explanation of the verification decision.
    """
    claim: str
    groundedness: GroundednessLevel
    citations: List[Citation]
    confidence: float
    reasoning: str
    
    @property
    def is_valid(self) -> bool:
        """Check if the claim is sufficiently grounded."""
        return self.groundedness == GroundednessLevel.GROUNDED
    
    @property
    def citation_count(self) -> int:
        """Number of supporting citations."""
        return len(self.citations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "claim": self.claim,
            "groundedness": self.groundedness.value,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "is_valid": self.is_valid
        }


@dataclass
class EvidenceBudget:
    """
    Manages evidence requirements for grounded responses.
    
    Enforces minimum citation requirements and tracks evidence quality
    to ensure AI responses are properly grounded in source data.
    
    Attributes:
        min_citations: Minimum citations required for a claim.
        min_relevance: Minimum relevance score for valid citations.
        require_multiple_sources: Require citations from different posts.
        max_age_days: Maximum age of citations in days (None = no limit).
    """
    min_citations: int = 2
    min_relevance: float = 0.3
    require_multiple_sources: bool = True
    max_age_days: Optional[int] = None
    
    _collected_citations: List[Citation] = field(default_factory=list)
    _verified_claims: List[ClaimVerificationResult] = field(default_factory=list)
    
    def __post_init__(self):
        self._collected_citations = []
        self._verified_claims = []
    
    def add_citation(self, citation: Citation) -> bool:
        """
        Add a citation to the evidence pool.
        
        Args:
            citation: Citation to add.
            
        Returns:
            True if citation meets quality requirements.
        """
        if citation.relevance_score < self.min_relevance:
            return False
        
        if self.max_age_days and citation.datetime:
            age = (datetime.now() - citation.datetime).days
            if age > self.max_age_days:
                return False
        
        self._collected_citations.append(citation)
        return True
    
    def validate_claim(
        self,
        claim: str,
        supporting_citations: List[Citation]
    ) -> ClaimVerificationResult:
        """
        Validate a claim against provided citations.
        
        Args:
            claim: The claim to validate.
            supporting_citations: Citations supporting the claim.
            
        Returns:
            ClaimVerificationResult with groundedness assessment.
        """
        valid_citations = [
            c for c in supporting_citations
            if c.relevance_score >= self.min_relevance
        ]
        
        if self.max_age_days:
            valid_citations = [
                c for c in valid_citations
                if not c.datetime or (datetime.now() - c.datetime).days <= self.max_age_days
            ]
        
        if len(valid_citations) == 0:
            result = ClaimVerificationResult(
                claim=claim,
                groundedness=GroundednessLevel.REJECTED,
                citations=[],
                confidence=0.0,
                reasoning="No valid citations found to support this claim."
            )
        elif len(valid_citations) < self.min_citations:
            result = ClaimVerificationResult(
                claim=claim,
                groundedness=GroundednessLevel.INSUFFICIENT_EVIDENCE,
                citations=valid_citations,
                confidence=0.3,
                reasoning=f"Only {len(valid_citations)} citation(s) found, "
                         f"minimum {self.min_citations} required."
            )
        else:
            if self.require_multiple_sources:
                unique_posts = set(c.post_id for c in valid_citations)
                if len(unique_posts) < 2:
                    result = ClaimVerificationResult(
                        claim=claim,
                        groundedness=GroundednessLevel.SPECULATION,
                        citations=valid_citations,
                        confidence=0.5,
                        reasoning="All citations come from a single source. "
                                 "Multiple independent sources required."
                    )
                else:
                    avg_relevance = sum(c.relevance_score for c in valid_citations) / len(valid_citations)
                    result = ClaimVerificationResult(
                        claim=claim,
                        groundedness=GroundednessLevel.GROUNDED,
                        citations=valid_citations,
                        confidence=min(0.95, avg_relevance + 0.2),
                        reasoning=f"Claim supported by {len(valid_citations)} citations "
                                 f"from {len(unique_posts)} unique sources."
                    )
            else:
                avg_relevance = sum(c.relevance_score for c in valid_citations) / len(valid_citations)
                result = ClaimVerificationResult(
                    claim=claim,
                    groundedness=GroundednessLevel.GROUNDED,
                    citations=valid_citations,
                    confidence=min(0.9, avg_relevance + 0.1),
                    reasoning=f"Claim supported by {len(valid_citations)} citations."
                )
        
        self._verified_claims.append(result)
        return result
    
    def get_evidence_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected evidence and verification results.
        
        Returns:
            Dictionary with evidence statistics.
        """
        grounded = sum(1 for v in self._verified_claims if v.is_valid)
        total = len(self._verified_claims)
        
        return {
            "total_citations": len(self._collected_citations),
            "verified_claims": total,
            "grounded_claims": grounded,
            "groundedness_rate": grounded / total if total > 0 else 0.0,
            "avg_citations_per_claim": (
                sum(v.citation_count for v in self._verified_claims) / total
                if total > 0 else 0.0
            ),
            "unique_sources": len(set(c.post_id for c in self._collected_citations))
        }
    
    def reset(self) -> None:
        """Clear all collected citations and verification results."""
        self._collected_citations = []
        self._verified_claims = []
    
    def should_refuse_response(self) -> tuple[bool, str]:
        """
        Determine if the response should be refused due to insufficient evidence.
        
        Returns:
            Tuple of (should_refuse, reason).
        """
        if len(self._collected_citations) == 0:
            return True, "No evidence found to support any claims."
        
        if self._verified_claims:
            grounded = sum(1 for v in self._verified_claims if v.is_valid)
            if grounded == 0:
                return True, "None of the claims could be adequately grounded in evidence."
        
        return False, ""
