from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from embedding_router import EmbeddingRouter, EmbeddingRouterDecision
from correction_cache import CorrectionCache
from cloud_fallback import CloudFallbackRouter, CloudFallbackDecision


@dataclass(frozen=True)
class HybridRouterDecision:
    """
    Immutable decision result from the HybridRouter.

    Attributes:
        input_text: Original user input.
        intent: Resolved intent string (or "unknown").
        accepted: Whether the intent was confidently recognized.
        source: Which layer resolved the intent (embedding_router, cloud_fallback, cache, ...).
        confidence: Confidence score of the decision.
        latency_s: Total classification time in seconds.
        fallback_used: Whether a cloud fallback was invoked.
        cache_hit: Whether the intent came from the correction cache.
        embedding_decision: The raw EmbeddingRouter decision (if available).
        fallback_decision: The raw CloudFallbackRouter decision (if available).
        metadata: Additional diagnostic metadata.
    """
    input_text: str
    intent: str
    accepted: bool
    source: str
    confidence: float
    latency_s: float
    fallback_used: bool = False
    cache_hit: bool = False
    embedding_decision: EmbeddingRouterDecision | None = None
    fallback_decision: CloudFallbackDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Optional: Fast-Path Heuristics ──────────────
# These detect obvious greeting/closing signals
# to avoid unnecessary embedding computation.
_SMALLTALK_SIGNALS = [
    "hallo", "hi ", "hey", "guten morgen", "guten abend", "guten tag",
    "tschüss", "bye", "bis morgen", "bis später", "bis bald",
    "danke", "dankeschön", "vielen dank",
    "wie geht", "was machst", "gut und dir", "freut mich",
    "ich bin wieder da", "da bin ich", "servus", "moin",
    "na du", "huhu", "aha", "okay", "ok ", "alles klar",
]


class HybridRouter:
    """
    Hybrid intent router combining local embeddings with cloud fallback.

    Routing strategy:
    1. Fast-Path: Detect obvious smalltalk signals without computation.
    2. Cache: Check exact-match correction cache (SQLite).
    3. Local: Run local EmbeddingRouter (SentenceTransformer + cosine similarity).
    4. Fallback: If local router is uncertain, invoke cloud fallback.
    5. Learn: Persist cloud corrections back to the cache for future hits.

    This design minimizes cloud costs by handling ~90% of routing
    decisions locally. The cloud fallback only runs for edge cases.
    """

    def __init__(
        self,
        embedding_router: EmbeddingRouter | None = None,
        cloud_fallback: CloudFallbackRouter | None = None,
        correction_cache: CorrectionCache | None = None,
        enable_fast_path: bool = True,
        enable_cache: bool = True,
        enable_fallback: bool = True,
    ):
        self.embedding_router = embedding_router or EmbeddingRouter()
        self.cloud_fallback = cloud_fallback
        self.correction_cache = correction_cache or CorrectionCache()
        self.enable_fast_path = enable_fast_path
        self.enable_cache = enable_cache
        self.enable_fallback = enable_fallback

    # ── Public API ────────────────────────────────

    def classify(self, text: str) -> HybridRouterDecision:
        """
        Classify user input into one of the configured intents.

        Args:
            text: User input text to classify.

        Returns:
            HybridRouterDecision with resolved intent and full diagnostic chain.
        """
        start = time.perf_counter()

        # ── Stage 1: Fast-Path ────────────────────
        if self.enable_fast_path:
            fast_decision = self._try_fast_path(text, start)
            if fast_decision is not None:
                return fast_decision

        # ── Stage 2: Cache Lookup ─────────────────
        if self.enable_cache:
            cache_decision = self._try_cache(text, start)
            if cache_decision is not None:
                return cache_decision

        # ── Stage 3: Local Embedding Router ───────
        embedding_decision = self.embedding_router.classify(text)

        if embedding_decision.accepted and embedding_decision.intent != "unknown":
            return HybridRouterDecision(
                input_text=text,
                intent=embedding_decision.intent,
                accepted=True,
                source="embedding_router",
                confidence=embedding_decision.confidence,
                latency_s=round(time.perf_counter() - start, 4),
                fallback_used=False,
                cache_hit=False,
                embedding_decision=embedding_decision,
                metadata={
                    "embedding_predicted_intent": embedding_decision.predicted_intent,
                    "embedding_confidence": embedding_decision.confidence,
                    "embedding_margin": embedding_decision.margin,
                    "cache_checked": self.enable_cache,
                    "fallback_available": self.enable_fallback and self.cloud_fallback is not None,
                },
            )

        # ── Stage 4: Cloud Fallback ───────────────
        if not self.enable_fallback or self.cloud_fallback is None:
            return self._unknown(
                text=text,
                start=start,
                embedding_decision=embedding_decision,
                reason="fallback_unavailable",
            )

        try:
            fallback_decision = self.cloud_fallback.classify(text)
        except Exception as exc:
            return self._unknown(
                text=text,
                start=start,
                embedding_decision=embedding_decision,
                reason="fallback_classification_failed",
                extra={"exception": str(exc)},
            )

        if fallback_decision.accepted and fallback_decision.intent != "unknown":
            # Learn from the correction
            self._save_correction(text, embedding_decision, fallback_decision)

            return HybridRouterDecision(
                input_text=text,
                intent=fallback_decision.intent,
                accepted=True,
                source="cloud_fallback",
                confidence=fallback_decision.confidence,
                latency_s=round(time.perf_counter() - start, 4),
                fallback_used=True,
                cache_hit=False,
                embedding_decision=embedding_decision,
                fallback_decision=fallback_decision,
                metadata={
                    "embedding_predicted_intent": embedding_decision.predicted_intent,
                    "embedding_confidence": embedding_decision.confidence,
                    "embedding_margin": embedding_decision.margin,
                    "fallback_model": fallback_decision.metadata.get("model"),
                },
            )

        return self._unknown(
            text=text,
            start=start,
            embedding_decision=embedding_decision,
            fallback_decision=fallback_decision,
            reason="fallback_returned_unknown",
        )

    # ── Internal Stages ──────────────────────────

    def _try_fast_path(
        self,
        text: str,
        start: float,
    ) -> HybridRouterDecision | None:
        """Check for obvious smalltalk signals without computation."""
        normalized = text.lower().strip()
        if any(signal in normalized for signal in _SMALLTALK_SIGNALS):
            return HybridRouterDecision(
                input_text=text,
                intent="smalltalk",
                accepted=True,
                source="fast_path",
                confidence=1.0,
                latency_s=round(time.perf_counter() - start, 4),
                fallback_used=False,
                cache_hit=False,
                metadata={"reason": "direct_smalltalk_signal"},
            )
        return None

    def _try_cache(
        self,
        text: str,
        start: float,
    ) -> HybridRouterDecision | None:
        """Check the correction cache for an exact match."""
        try:
            cached_intent = self.correction_cache.get_intent(text)
        except Exception:
            return None

        if cached_intent is not None:
            # Register a hit for analytics
            try:
                self.correction_cache.register_hit(text)
            except Exception:
                pass

            return HybridRouterDecision(
                input_text=text,
                intent=cached_intent,
                accepted=True,
                source="correction_cache",
                confidence=1.0,
                latency_s=round(time.perf_counter() - start, 4),
                fallback_used=False,
                cache_hit=True,
                metadata={"cache_source": "exact_hash"},
            )

        return None

    def _save_correction(
        self,
        text: str,
        embedding_decision: EmbeddingRouterDecision,
        fallback_decision: CloudFallbackDecision,
    ) -> None:
        """Persist a cloud fallback correction to the cache."""
        if not self.enable_cache:
            return

        try:
            self.correction_cache.save_correction(
                text=text,
                final_intent=fallback_decision.intent,
                local_prediction=embedding_decision.predicted_intent,
                local_confidence=embedding_decision.confidence,
                fallback_intent=fallback_decision.intent,
                fallback_confidence=fallback_decision.confidence,
            )
        except Exception:
            pass

    def _unknown(
        self,
        text: str,
        start: float,
        embedding_decision: EmbeddingRouterDecision | None = None,
        fallback_decision: CloudFallbackDecision | None = None,
        reason: str = "unknown",
        extra: dict[str, Any] | None = None,
    ) -> HybridRouterDecision:
        """Return a rejected/unknown decision."""
        metadata: dict[str, Any] = {"reason": reason}
        if embedding_decision:
            metadata.update({
                "embedding_predicted_intent": embedding_decision.predicted_intent,
                "embedding_confidence": embedding_decision.confidence,
                "embedding_margin": embedding_decision.margin,
            })
        if extra:
            metadata.update(extra)

        return HybridRouterDecision(
            input_text=text,
            intent="unknown",
            accepted=False,
            source="hybrid_router",
            confidence=0.0,
            latency_s=round(time.perf_counter() - start, 4),
            fallback_used=fallback_decision is not None,
            cache_hit=False,
            embedding_decision=embedding_decision,
            fallback_decision=fallback_decision,
            metadata=metadata,
        )
