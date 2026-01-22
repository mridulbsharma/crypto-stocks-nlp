"""
Self-Socratic Refinement loop for CryptoSentinel.

Implements the Plan → Execute → Verify → Finalize reasoning pattern
for producing grounded, evidence-based responses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Callable

from ..context_engine.retriever import PostRetriever, SearchResult
from ..context_engine.evidence import Citation, EvidenceBudget, ClaimVerificationResult
from .personas import DomainAnalyst, EvidenceVerifier, VerificationResult, VerificationStatus


class RefinementStage(Enum):
    """Stages of the Socratic refinement process."""
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    REFUSED = "refused"


@dataclass
class SubQuestion:
    """A sub-question derived from the main query."""
    question: str
    tool_needed: str
    priority: int = 1
    answered: bool = False
    answer: Optional[str] = None
    evidence: List[Citation] = field(default_factory=list)


@dataclass
class Plan:
    """
    Execution plan for answering a question.
    
    Attributes:
        original_question: The user's original question.
        sub_questions: Decomposed sub-questions.
        tools_required: Tools needed to execute the plan.
        estimated_sources: Estimated number of sources needed.
    """
    original_question: str
    sub_questions: List[SubQuestion]
    tools_required: List[str]
    estimated_sources: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "original_question": self.original_question,
            "sub_questions": [
                {"question": sq.question, "tool": sq.tool_needed, "answered": sq.answered}
                for sq in self.sub_questions
            ],
            "tools_required": self.tools_required,
            "estimated_sources": self.estimated_sources
        }


@dataclass
class Verification:
    """
    Results from the verification stage.
    
    Attributes:
        claims_verified: Number of verified claims.
        claims_rejected: Number of rejected claims.
        gaps_found: Identified gaps in evidence.
        contradictions: Found contradictions.
        overall_confidence: Overall confidence score.
    """
    claims_verified: int
    claims_rejected: int
    gaps_found: List[str]
    contradictions: List[str]
    overall_confidence: float
    verification_results: List[VerificationResult] = field(default_factory=list)
    
    @property
    def is_acceptable(self) -> bool:
        """Check if verification results are acceptable."""
        if self.claims_verified + self.claims_rejected == 0:
            return False
        rate = self.claims_verified / (self.claims_verified + self.claims_rejected)
        return rate >= 0.5 and self.overall_confidence >= 0.5


@dataclass
class RefinedAnswer:
    """
    Final refined answer with full context.
    
    Attributes:
        question: Original question.
        answer: Final answer text.
        citations: Supporting citations.
        confidence: Overall confidence.
        groundedness_score: How well-grounded the answer is.
        verification: Verification details.
        trace: Full reasoning trace.
        refused: Whether the answer was refused due to insufficient evidence.
        refusal_reason: Reason for refusal if applicable.
    """
    question: str
    answer: str
    citations: List[Citation]
    confidence: float
    groundedness_score: float
    verification: Verification
    trace: List[Dict[str, Any]]
    refused: bool = False
    refusal_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "question": self.question,
            "answer": self.answer if not self.refused else f"[REFUSED] {self.refusal_reason}",
            "citations": [c.to_dict() for c in self.citations],
            "confidence": round(self.confidence, 3),
            "groundedness_score": round(self.groundedness_score, 3),
            "claims_verified": self.verification.claims_verified,
            "claims_rejected": self.verification.claims_rejected,
            "refused": self.refused
        }
    
    def format_with_citations(self) -> str:
        """Format the answer with inline citations."""
        if self.refused:
            return f"⚠️ **Unable to provide a grounded response.**\n\n{self.refusal_reason}"
        
        lines = [
            f"## Answer",
            "",
            self.answer,
            "",
            f"**Confidence**: {self.confidence:.1%}",
            f"**Groundedness**: {self.groundedness_score:.1%}",
            "",
            "### Sources",
        ]
        
        for i, citation in enumerate(self.citations[:5], 1):
            lines.append(f"{i}. [{citation.subreddit}]({citation.url}) - {citation.text_snippet[:100]}...")
        
        return "\n".join(lines)


class SocraticRefiner:
    """
    Self-Socratic Refinement engine.
    
    Implements the Plan → Execute → Verify → Finalize loop for producing
    grounded, evidence-based answers to questions about crypto/stock discussions.
    """
    
    def __init__(
        self,
        retriever: PostRetriever,
        evidence_budget: Optional[EvidenceBudget] = None
    ):
        """
        Initialize the Socratic Refiner.
        
        Args:
            retriever: PostRetriever for searching posts.
            evidence_budget: Evidence requirements (uses default if None).
        """
        self.retriever = retriever
        self.evidence_budget = evidence_budget or EvidenceBudget()
        self.analyst = DomainAnalyst()
        self.verifier = EvidenceVerifier()
        
        self._trace: List[Dict[str, Any]] = []
        self._current_stage = RefinementStage.PLANNING
    
    def refine(self, question: str) -> RefinedAnswer:
        """
        Execute the full refinement loop for a question.
        
        Args:
            question: User's question.
            
        Returns:
            RefinedAnswer with grounded response or refusal.
        """
        self._trace = []
        self._current_stage = RefinementStage.PLANNING
        
        self._log("Starting Socratic refinement", {"question": question})
        
        plan = self.plan(question)
        self._log("Plan created", plan.to_dict())
        
        self._current_stage = RefinementStage.EXECUTING
        evidence, sub_answers = self.execute(plan)
        self._log("Execution complete", {
            "citations_collected": len(evidence),
            "sub_questions_answered": sum(1 for sq in plan.sub_questions if sq.answered)
        })
        
        self._current_stage = RefinementStage.VERIFYING
        verification = self.verify(evidence, sub_answers)
        self._log("Verification complete", {
            "verified": verification.claims_verified,
            "rejected": verification.claims_rejected,
            "acceptable": verification.is_acceptable
        })
        
        self._current_stage = RefinementStage.FINALIZING
        answer = self.finalize(question, evidence, verification, sub_answers)
        
        self._current_stage = RefinementStage.COMPLETE if not answer.refused else RefinementStage.REFUSED
        self._log("Refinement complete", {"refused": answer.refused})
        
        return answer
    
    def plan(self, question: str) -> Plan:
        """
        Create an execution plan by decomposing the question.
        
        Args:
            question: Original question.
            
        Returns:
            Plan with sub-questions and required tools.
        """
        sub_questions = []
        tools_required = set()
        
        q_lower = question.lower()
        
        sub_questions.append(SubQuestion(
            question=f"What posts are relevant to: {question}",
            tool_needed="reddit_search",
            priority=1
        ))
        tools_required.add("reddit_search")
        
        if any(word in q_lower for word in ['sentiment', 'feel', 'opinion', 'bullish', 'bearish']):
            sub_questions.append(SubQuestion(
                question="What is the overall sentiment?",
                tool_needed="sentiment_aggregate",
                priority=2
            ))
            tools_required.add("sentiment_aggregate")
        
        if any(word in q_lower for word in ['trend', 'topic', 'discuss', 'talking']):
            sub_questions.append(SubQuestion(
                question="What are the main topic clusters?",
                tool_needed="topic_clusters",
                priority=2
            ))
            tools_required.add("topic_clusters")
        
        if any(word in q_lower for word in ['crypto', 'stock', 'type', 'category']):
            sub_questions.append(SubQuestion(
                question="Is this crypto or stock related?",
                tool_needed="classify_post",
                priority=3
            ))
            tools_required.add("classify_post")
        
        sub_questions.append(SubQuestion(
            question=f"Generate a grounded brief about: {question}",
            tool_needed="generate_brief",
            priority=10
        ))
        tools_required.add("generate_brief")
        
        sub_questions.sort(key=lambda x: x.priority)
        
        return Plan(
            original_question=question,
            sub_questions=sub_questions,
            tools_required=list(tools_required),
            estimated_sources=max(5, len(sub_questions) * 2)
        )
    
    def execute(self, plan: Plan) -> tuple[List[Citation], List[Dict[str, Any]]]:
        """
        Execute the plan by calling required tools.
        
        Args:
            plan: Execution plan.
            
        Returns:
            Tuple of (collected_citations, sub_question_answers).
        """
        all_citations = []
        sub_answers = []
        
        for sq in plan.sub_questions:
            try:
                if sq.tool_needed == "reddit_search":
                    results, citations = self.retriever.search_with_citations(
                        plan.original_question,
                        k=10
                    )
                    sq.evidence = citations
                    sq.answer = f"Found {len(results)} relevant posts"
                    sq.answered = True
                    all_citations.extend(citations)
                    
                    sub_answers.append({
                        "question": sq.question,
                        "answer": sq.answer,
                        "sources": len(citations)
                    })
                
                elif sq.tool_needed == "sentiment_aggregate":
                    if all_citations:
                        positive = sum(1 for c in all_citations if hasattr(c, 'score') and c.score > 0)
                        sq.answer = f"Sentiment: {positive}/{len(all_citations)} positive engagement"
                        sq.answered = True
                        sub_answers.append({
                            "question": sq.question,
                            "answer": sq.answer
                        })
                
                elif sq.tool_needed == "generate_brief":
                    analysis = self.analyst.analyze(
                        plan.original_question,
                        all_citations
                    )
                    sq.answer = self.analyst.format_analysis(analysis)
                    sq.answered = True
                    sub_answers.append({
                        "question": sq.question,
                        "answer": sq.answer,
                        "analysis": analysis
                    })
                
            except Exception as e:
                self._log("Tool execution error", {
                    "tool": sq.tool_needed,
                    "error": str(e)
                })
        
        return all_citations, sub_answers
    
    def verify(
        self,
        evidence: List[Citation],
        sub_answers: List[Dict[str, Any]]
    ) -> Verification:
        """
        Verify the gathered evidence and identify gaps.
        
        Args:
            evidence: Collected citations.
            sub_answers: Answers to sub-questions.
            
        Returns:
            Verification result with gaps and contradictions.
        """
        claims = []
        for sa in sub_answers:
            if 'answer' in sa and sa['answer']:
                claims.append(sa['answer'])
        
        verification_results = []
        for claim in claims:
            result = self.verifier.verify(claim, evidence)
            verification_results.append(result)
        
        verified = sum(1 for r in verification_results if r.is_valid)
        rejected = sum(1 for r in verification_results if r.status == VerificationStatus.REJECTED)
        
        gaps = []
        if len(evidence) < 3:
            gaps.append("Limited evidence available (fewer than 3 sources)")
        if len(set(c.subreddit for c in evidence)) < 2:
            gaps.append("Evidence from single subreddit only")
        
        contradictions = []
        
        if evidence:
            avg_relevance = sum(c.relevance_score for c in evidence) / len(evidence)
        else:
            avg_relevance = 0.0
        
        return Verification(
            claims_verified=verified,
            claims_rejected=rejected,
            gaps_found=gaps,
            contradictions=contradictions,
            overall_confidence=avg_relevance,
            verification_results=verification_results
        )
    
    def finalize(
        self,
        question: str,
        evidence: List[Citation],
        verification: Verification,
        sub_answers: List[Dict[str, Any]]
    ) -> RefinedAnswer:
        """
        Produce the final refined answer.
        
        Args:
            question: Original question.
            evidence: Collected evidence.
            verification: Verification results.
            sub_answers: Sub-question answers.
            
        Returns:
            RefinedAnswer with final response or refusal.
        """
        should_refuse, refusal_reason = self.evidence_budget.should_refuse_response()
        
        if should_refuse or not verification.is_acceptable:
            return RefinedAnswer(
                question=question,
                answer="",
                citations=evidence,
                confidence=0.0,
                groundedness_score=0.0,
                verification=verification,
                trace=self._trace,
                refused=True,
                refusal_reason=refusal_reason or "Insufficient evidence to provide a grounded response. "
                               f"Verification rate: {verification.claims_verified}/{verification.claims_verified + verification.claims_rejected}"
            )
        
        answer_parts = []
        for sa in sub_answers:
            if 'analysis' in sa:
                answer_parts.append(sa.get('answer', ''))
        
        if not answer_parts:
            answer_parts.append(f"Based on {len(evidence)} sources from Reddit discussions:")
            for sa in sub_answers[:3]:
                if sa.get('answer'):
                    answer_parts.append(f"• {sa['answer']}")
        
        final_answer = "\n\n".join(answer_parts)
        
        groundedness = verification.claims_verified / max(1, verification.claims_verified + verification.claims_rejected)
        
        return RefinedAnswer(
            question=question,
            answer=final_answer,
            citations=evidence,
            confidence=verification.overall_confidence,
            groundedness_score=groundedness,
            verification=verification,
            trace=self._trace,
            refused=False
        )
    
    def _log(self, event: str, data: Dict[str, Any]) -> None:
        """Add entry to reasoning trace."""
        self._trace.append({
            "timestamp": datetime.now().isoformat(),
            "stage": self._current_stage.value,
            "event": event,
            "data": data
        })
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get the full reasoning trace."""
        return self._trace.copy()
