from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CloudFallbackDecision:
    """
    Immutable decision result from a cloud-based fallback router.

    Attributes:
        input_text: Original user input.
        intent: Resolved intent string (or "unknown").
        confidence: Confidence score of the decision.
        accepted: Whether the intent was confidently recognized.
        latency_s: Classification time in seconds.
        raw_response: Raw response from the cloud provider.
        metadata: Additional diagnostic information (model name, etc.).
    """
    input_text: str
    intent: str
    confidence: float
    accepted: bool
    latency_s: float
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class CloudFallbackRouter:
    """
    Abstract base class for cloud-based intent classification fallback.

    Subclasses implement the actual API call to a specific cloud provider
    (e.g., Gemini, OpenAI, Claude). The HybridRouter uses this interface
    so it remains provider-agnostic.

    To use the HybridRouter with a cloud fallback, implement a subclass
    and pass it to the HybridRouter constructor:

        class MyGeminiRouter(CloudFallbackRouter):
            def classify(self, text: str) -> CloudFallbackDecision:
                # ... call Gemini API ...
                pass

        router = HybridRouter(
            cloud_fallback=MyGeminiRouter()
        )
    """

    def classify(self, text: str) -> CloudFallbackDecision:
        """
        Classify user input using a cloud-based intent classifier.

        Args:
            text: User input text to classify.

        Returns:
            CloudFallbackDecision with the resolved intent.
        """
        raise NotImplementedError(
            "Subclasses must implement classify()."
        )
