"""
Dual-persona system for CryptoSentinel reasoning.

Implements DomainAnalyst (creative analysis) and EvidenceVerifier (strict verification)
personas that work together to produce grounded, reliable insights.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

from ..context_engine.evidence import Citation, GroundednessLevel


class VerificationStatus(Enum):
    """Status of a verification decision."""
    APPROVED = "approved"
    SPECULATION_FLAGGED = "speculation_flagged"
    REJECTED = "rejected"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    VETOED = "vetoed"


@dataclass
class VerificationResult:
    """
    Result from the EvidenceVerifier persona.
    
    Attributes:
        status: Verification status.
        claim: Original claim text.
        citations_used: Citations supporting the claim.
        reasoning: Explanation of the verification decision.
        confidence: Confidence in the verification (0-1).
        veto_applied: Whether the verifier applied veto power.
        suggested_revision: Suggested text revision if needed.
    """
    status: VerificationStatus
    claim: str
    citations_used: List[Citation]
    reasoning: str
    confidence: float
    veto_applied: bool = False
    suggested_revision: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if the claim passed verification."""
        return self.status == VerificationStatus.APPROVED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "claim": self.claim,
            "citations_count": len(self.citations_used),
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 3),
            "veto_applied": self.veto_applied,
            "is_valid": self.is_valid
        }


class DomainAnalyst:
    """
    Creative but bounded analysis persona.
    
    Responsible for generating insights and hypotheses about crypto/stock trends
    while clearly labeling any speculation.
    """
    
    SYSTEM_PROMPT = """You are a Domain Analyst specializing in cryptocurrency and stock market trends.
    
Your role is to:
1. Analyze Reddit discussions from r/wallstreetbets and r/CryptoMoonShots
2. Identify emerging trends, sentiment patterns, and notable discussions
3. Generate insights based on the evidence provided

IMPORTANT CONSTRAINTS:
- You MAY hypothesize about trends, but you MUST clearly label speculation
- Always distinguish between "evidence shows" and "this suggests"
- Never make definitive claims without supporting citations
- Use hedging language for uncertain conclusions

Output format:
- State observations with [GROUNDED] or [SPECULATION] tags
- Include citation references for grounded claims
- Provide confidence level (high/medium/low) for each insight
"""
    
    def __init__(self, min_confidence_for_grounded: float = 0.7):
        """
        Initialize the Domain Analyst.
        
        Args:
            min_confidence_for_grounded: Minimum confidence to label as grounded.
        """
        self.min_confidence = min_confidence_for_grounded
    
    def analyze(
        self,
        query: str,
        evidence: List[Citation],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate analysis based on query and evidence.
        
        Args:
            query: User's question or topic.
            evidence: List of supporting citations.
            context: Additional context.
            
        Returns:
            Analysis result with insights and labels.
        """
        if not evidence:
            return {
                "status": "insufficient_evidence",
                "insights": [],
                "recommendation": "Cannot provide analysis without supporting evidence."
            }
        
        avg_relevance = sum(c.relevance_score for c in evidence) / len(evidence)
        unique_sources = len(set(c.post_id for c in evidence))
        
        groundedness = "grounded" if (
            avg_relevance >= self.min_confidence and unique_sources >= 2
        ) else "speculation"
        
        insights = self._extract_insights(query, evidence)
        
        return {
            "status": "analyzed",
            "groundedness": groundedness,
            "avg_relevance": avg_relevance,
            "source_count": unique_sources,
            "insights": insights,
            "evidence_summary": f"Based on {len(evidence)} citations from {unique_sources} sources",
            "confidence": "high" if avg_relevance > 0.7 else "medium" if avg_relevance > 0.4 else "low"
        }
    
    def _extract_insights(
        self,
        query: str,
        evidence: List[Citation]
    ) -> List[Dict[str, Any]]:
        """Extract key insights from evidence."""
        insights = []
        
        subreddit_counts = {}
        for c in evidence:
            subreddit_counts[c.subreddit] = subreddit_counts.get(c.subreddit, 0) + 1
        
        if subreddit_counts:
            dominant = max(subreddit_counts, key=subreddit_counts.get)
            insights.append({
                "type": "source_distribution",
                "label": "[GROUNDED]",
                "content": f"Discussion primarily found in r/{dominant}",
                "evidence": [c.post_id for c in evidence if c.subreddit == dominant][:3]
            })
        
        high_score = [c for c in evidence if c.score > 100]
        if high_score:
            insights.append({
                "type": "engagement",
                "label": "[GROUNDED]",
                "content": f"Topic has high engagement ({len(high_score)} highly-voted posts)",
                "evidence": [c.post_id for c in high_score[:3]]
            })
        
        if len(set(c.subreddit for c in evidence)) > 1:
            insights.append({
                "type": "cross_community",
                "label": "[SPECULATION]",
                "content": "Interest spans both crypto and stock communities, suggesting broader market relevance",
                "confidence": "medium"
            })
        
        return insights
    
    def format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis for output."""
        if analysis["status"] == "insufficient_evidence":
            return f"⚠️ {analysis['recommendation']}"
        
        lines = [
            f"## Analysis Summary",
            f"**Confidence**: {analysis['confidence'].upper()}",
            f"**Groundedness**: {analysis['groundedness'].upper()}",
            f"**Sources**: {analysis['evidence_summary']}",
            "",
            "### Key Insights"
        ]
        
        for insight in analysis["insights"]:
            label = insight.get("label", "[ANALYSIS]")
            lines.append(f"- {label} {insight['content']}")
        
        return "\n".join(lines)


class EvidenceVerifier:
    """
    Strict verification persona with veto power.
    
    Responsible for verifying claims against evidence and rejecting
    any claim that lacks proper citation support.
    """
    
    SYSTEM_PROMPT = """You are an Evidence Verifier with strict verification standards.

Your role is to:
1. Verify every claim against provided evidence
2. REJECT any claim without proper citation
3. Flag speculation that isn't properly labeled
4. Exercise VETO power on responses that don't meet evidence standards

VERIFICATION RULES:
- Every factual claim MUST have at least 1 supporting citation
- Claims about trends MUST have 2+ independent sources
- Speculation MUST be clearly labeled as such
- You CAN veto the entire response if evidence standards aren't met

You are the last line of defense against hallucination and misinformation.
"""
    
    def __init__(
        self,
        min_citations: int = 1,
        min_citations_for_trends: int = 2,
        strict_mode: bool = True
    ):
        """
        Initialize the Evidence Verifier.
        
        Args:
            min_citations: Minimum citations for basic claims.
            min_citations_for_trends: Minimum citations for trend claims.
            strict_mode: Whether to apply strict verification.
        """
        self.min_citations = min_citations
        self.min_citations_for_trends = min_citations_for_trends
        self.strict_mode = strict_mode
    
    def verify(
        self,
        claim: str,
        evidence: List[Citation]
    ) -> VerificationResult:
        """
        Verify a claim against provided evidence.
        
        Args:
            claim: The claim to verify.
            evidence: List of supporting citations.
            
        Returns:
            VerificationResult with verification decision.
        """
        if not evidence:
            return VerificationResult(
                status=VerificationStatus.REJECTED,
                claim=claim,
                citations_used=[],
                reasoning="No citations provided to support this claim.",
                confidence=0.0,
                veto_applied=self.strict_mode
            )
        
        relevant_citations = [c for c in evidence if c.relevance_score >= 0.3]
        
        if len(relevant_citations) == 0:
            return VerificationResult(
                status=VerificationStatus.REJECTED,
                claim=claim,
                citations_used=[],
                reasoning="No citations meet minimum relevance threshold (0.3).",
                confidence=0.0
            )
        
        is_trend_claim = any(word in claim.lower() for word in [
            'trend', 'increasing', 'growing', 'declining', 'popular',
            'sentiment', 'majority', 'most', 'generally'
        ])
        
        required = self.min_citations_for_trends if is_trend_claim else self.min_citations
        
        if len(relevant_citations) < required:
            return VerificationResult(
                status=VerificationStatus.NEEDS_MORE_EVIDENCE,
                claim=claim,
                citations_used=relevant_citations,
                reasoning=f"Found {len(relevant_citations)} citations, but {required} required for this type of claim.",
                confidence=len(relevant_citations) / required * 0.5,
                suggested_revision=f"Consider qualifying this as speculation or gathering more evidence."
            )
        
        unique_sources = set(c.post_id for c in relevant_citations)
        if is_trend_claim and len(unique_sources) < 2:
            return VerificationResult(
                status=VerificationStatus.SPECULATION_FLAGGED,
                claim=claim,
                citations_used=relevant_citations,
                reasoning="Trend claims require multiple independent sources. All citations from single source.",
                confidence=0.4,
                suggested_revision="Label this as speculation: 'One source suggests...'"
            )
        
        avg_relevance = sum(c.relevance_score for c in relevant_citations) / len(relevant_citations)
        confidence = min(0.95, avg_relevance * 0.8 + 0.2)
        
        return VerificationResult(
            status=VerificationStatus.APPROVED,
            claim=claim,
            citations_used=relevant_citations,
            reasoning=f"Claim verified with {len(relevant_citations)} citations from {len(unique_sources)} sources.",
            confidence=confidence
        )
    
    def verify_response(
        self,
        claims: List[str],
        all_evidence: List[Citation]
    ) -> Dict[str, Any]:
        """
        Verify all claims in a response.
        
        Args:
            claims: List of claims to verify.
            all_evidence: All available evidence.
            
        Returns:
            Dictionary with overall verification result.
        """
        results = [self.verify(claim, all_evidence) for claim in claims]
        
        approved = sum(1 for r in results if r.is_valid)
        rejected = sum(1 for r in results if r.status == VerificationStatus.REJECTED)
        flagged = sum(1 for r in results if r.status == VerificationStatus.SPECULATION_FLAGGED)
        
        should_veto = self.strict_mode and (
            rejected > len(claims) / 2 or
            (approved == 0 and len(claims) > 0)
        )
        
        return {
            "total_claims": len(claims),
            "approved": approved,
            "rejected": rejected,
            "speculation_flagged": flagged,
            "veto_recommended": should_veto,
            "verification_rate": approved / len(claims) if claims else 0,
            "results": [r.to_dict() for r in results],
            "summary": self._generate_summary(results, should_veto)
        }
    
    def _generate_summary(
        self,
        results: List[VerificationResult],
        veto: bool
    ) -> str:
        """Generate verification summary."""
        if veto:
            return "⛔ VETO: Response does not meet evidence standards. Too many unverified claims."
        
        approved = sum(1 for r in results if r.is_valid)
        total = len(results)
        
        if approved == total:
            return f"✅ All {total} claims verified successfully."
        elif approved > total / 2:
            return f"⚠️ {approved}/{total} claims verified. Some claims flagged for review."
        else:
            return f"🔴 Only {approved}/{total} claims verified. Significant revision needed."
    
    def veto(self, reason: str) -> VerificationResult:
        """
        Apply veto power to reject a response entirely.
        
        Args:
            reason: Reason for the veto.
            
        Returns:
            VerificationResult with veto applied.
        """
        return VerificationResult(
            status=VerificationStatus.VETOED,
            claim="[ENTIRE RESPONSE]",
            citations_used=[],
            reasoning=reason,
            confidence=0.0,
            veto_applied=True
        )
