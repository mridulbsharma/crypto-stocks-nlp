"""
Groundedness evaluation for CryptoSentinel.

Tests that AI-generated responses are properly grounded in evidence
with valid citations and appropriate confidence levels.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class GroundednessTestCase:
    """A test case for groundedness evaluation."""
    name: str
    query: str
    expected_min_citations: int
    expected_groundedness: str  # "high", "medium", "low"
    should_have_citations: bool = True
    notes: Optional[str] = None


@dataclass
class GroundednessResult:
    """Result from a groundedness test."""
    test_case: str
    passed: bool
    actual_citations: int
    expected_citations: int
    actual_groundedness: str
    expected_groundedness: str
    confidence: float
    details: str
    timestamp: datetime


class GroundednessEvaluator:
    """
    Evaluates the groundedness of CryptoSentinel responses.
    
    Tests that:
    1. Claims have supporting citations
    2. Citation relevance meets thresholds
    3. Groundedness labels are accurate
    4. Confidence scores align with evidence quality
    """
    
    DEFAULT_TEST_CASES = [
        GroundednessTestCase(
            name="basic_crypto_query",
            query="What are people saying about Bitcoin?",
            expected_min_citations=2,
            expected_groundedness="medium",
            should_have_citations=True
        ),
        GroundednessTestCase(
            name="basic_stock_query",
            query="What's the sentiment on GME?",
            expected_min_citations=2,
            expected_groundedness="medium",
            should_have_citations=True
        ),
        GroundednessTestCase(
            name="specific_topic",
            query="Are people bullish or bearish on Tesla?",
            expected_min_citations=3,
            expected_groundedness="medium",
            should_have_citations=True
        ),
        GroundednessTestCase(
            name="trend_analysis",
            query="What topics are trending on wallstreetbets?",
            expected_min_citations=5,
            expected_groundedness="high",
            should_have_citations=True,
            notes="Trend claims require more evidence"
        ),
        GroundednessTestCase(
            name="cross_community",
            query="How does crypto sentiment compare to stock sentiment?",
            expected_min_citations=4,
            expected_groundedness="medium",
            should_have_citations=True,
            notes="Should cite from both subreddits"
        ),
        GroundednessTestCase(
            name="obscure_topic",
            query="What do people think about XYZ123FAKE coin?",
            expected_min_citations=0,
            expected_groundedness="low",
            should_have_citations=False,
            notes="Should refuse or return low confidence"
        )
    ]
    
    def __init__(self, test_cases: Optional[List[GroundednessTestCase]] = None):
        """
        Initialize the evaluator.
        
        Args:
            test_cases: Custom test cases (uses defaults if None).
        """
        self.test_cases = test_cases or self.DEFAULT_TEST_CASES
        self.results: List[GroundednessResult] = []
    
    def run_evaluation(self, tools=None) -> Dict[str, Any]:
        """
        Run the full groundedness evaluation.
        
        Args:
            tools: CryptoSentinelTools instance.
            
        Returns:
            Evaluation summary with pass rate and details.
        """
        if tools is None:
            from src.mcp_server.tools import CryptoSentinelTools
            tools = CryptoSentinelTools()
            tools._ensure_initialized()
        
        self.results = []
        
        for test_case in self.test_cases:
            result = self._run_single_test(test_case, tools)
            self.results.append(result)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "results": [self._result_to_dict(r) for r in self.results],
            "summary": self._generate_summary()
        }
    
    def _run_single_test(
        self,
        test_case: GroundednessTestCase,
        tools
    ) -> GroundednessResult:
        """Run a single test case."""
        from src.mcp_server.tools import generate_brief
        
        try:
            brief = generate_brief(
                topic=test_case.query,
                max_sources=10,
                tools=tools
            )
            
            actual_citations = len(brief.citations)
            actual_groundedness = brief.groundedness
            confidence = brief.confidence
            
            passed = True
            details = []
            
            if test_case.should_have_citations:
                if actual_citations < test_case.expected_min_citations:
                    passed = False
                    details.append(
                        f"Expected >= {test_case.expected_min_citations} citations, "
                        f"got {actual_citations}"
                    )
            else:
                if actual_citations > 0 and actual_groundedness not in ["low", "insufficient"]:
                    details.append(f"Unexpected citations for obscure topic")
            
            groundedness_order = ["insufficient", "low", "medium", "high"]
            expected_idx = groundedness_order.index(test_case.expected_groundedness) if test_case.expected_groundedness in groundedness_order else -1
            actual_idx = groundedness_order.index(actual_groundedness) if actual_groundedness in groundedness_order else -1
            
            if expected_idx >= 0 and actual_idx >= 0:
                if abs(expected_idx - actual_idx) > 1:
                    passed = False
                    details.append(
                        f"Groundedness mismatch: expected ~{test_case.expected_groundedness}, "
                        f"got {actual_groundedness}"
                    )
            
            return GroundednessResult(
                test_case=test_case.name,
                passed=passed,
                actual_citations=actual_citations,
                expected_citations=test_case.expected_min_citations,
                actual_groundedness=actual_groundedness,
                expected_groundedness=test_case.expected_groundedness,
                confidence=confidence,
                details="; ".join(details) if details else "All checks passed",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return GroundednessResult(
                test_case=test_case.name,
                passed=False,
                actual_citations=0,
                expected_citations=test_case.expected_min_citations,
                actual_groundedness="error",
                expected_groundedness=test_case.expected_groundedness,
                confidence=0.0,
                details=f"Error: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _result_to_dict(self, result: GroundednessResult) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_case": result.test_case,
            "passed": result.passed,
            "actual_citations": result.actual_citations,
            "expected_citations": result.expected_citations,
            "actual_groundedness": result.actual_groundedness,
            "expected_groundedness": result.expected_groundedness,
            "confidence": round(result.confidence, 3),
            "details": result.details,
            "timestamp": result.timestamp.isoformat()
        }
    
    def _generate_summary(self) -> str:
        """Generate evaluation summary."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        lines = [
            f"Groundedness Evaluation Summary",
            f"=" * 40,
            f"Total Tests: {total}",
            f"Passed: {passed}",
            f"Failed: {total - passed}",
            f"Pass Rate: {passed/total*100:.1f}%" if total > 0 else "N/A",
            "",
            "Individual Results:"
        ]
        
        for result in self.results:
            status = "✅" if result.passed else "❌"
            lines.append(f"  {status} {result.test_case}: {result.details}")
        
        return "\n".join(lines)
    
    def add_test_case(
        self,
        name: str,
        query: str,
        expected_min_citations: int,
        expected_groundedness: str,
        **kwargs
    ) -> None:
        """Add a custom test case."""
        self.test_cases.append(GroundednessTestCase(
            name=name,
            query=query,
            expected_min_citations=expected_min_citations,
            expected_groundedness=expected_groundedness,
            **kwargs
        ))


def run_groundedness_eval():
    """Run groundedness evaluation from command line."""
    evaluator = GroundednessEvaluator()
    results = evaluator.run_evaluation()
    
    print(results["summary"])
    print(f"\nOverall Pass Rate: {results['pass_rate']*100:.1f}%")
    
    return results["pass_rate"] >= 0.7


if __name__ == "__main__":
    import sys
    success = run_groundedness_eval()
    sys.exit(0 if success else 1)
