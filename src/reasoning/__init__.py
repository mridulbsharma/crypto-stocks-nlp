"""Reasoning Module - Self-Socratic refinement and dual-persona verification."""

from .socratic_refine import SocraticRefiner, RefinedAnswer, Plan, Verification
from .personas import DomainAnalyst, EvidenceVerifier, VerificationResult

__all__ = [
    "SocraticRefiner",
    "RefinedAnswer",
    "Plan",
    "Verification",
    "DomainAnalyst",
    "EvidenceVerifier",
    "VerificationResult",
]
