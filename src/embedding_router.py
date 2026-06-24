from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from embedding_provider import EmbeddingProvider
from sklearn.metrics.pairwise import cosine_similarity

# ── Default Config Path ──────────────────────────
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "intents.yaml"


@dataclass(frozen=True)
class EmbeddingRouterDecision:
    """Immutable decision result from the EmbeddingRouter."""

    input_text: str
    intent: str
    predicted_intent: str
    second_intent: str
    confidence: float
    second_score: float
    margin: float
    accepted: bool
    latency_s: float
    scores: dict[str, float]
    model_name: str
    source: str = "embedding_router"
    embedding: list[float] | None = None


class EmbeddingRouter:
    """
    Lightweight intent classifier using sentence embeddings + cosine similarity.

    Classifies user input into predefined intents by comparing its embedding
    against intent centroids (mean embeddings of prototype sentences).

    Architecture:
        SentenceTransformer → Cosine Similarity → Confidence Thresholds

    Features:
        - No LLM required: runs locally with minimal latency
        - Multilingual: works with any language supported by the SentenceTransformer
        - Configurable intents via external YAML file
        - Returns full scoring breakdown for transparency
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        model_name: str | None = None,
        accept_threshold: float | None = None,
        weak_accept_threshold: float | None = None,
        min_margin: float | None = None,
    ):
        # Load config (YAML or default)
        config = self._load_config(
            Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        )

        self.model_name = model_name or config.get("model", {}).get(
            "default",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

        # Thresholds: constructor args > YAML config > hardcoded defaults
        thresholds = config.get("thresholds", {})
        self.accept_threshold = (
            accept_threshold
            if accept_threshold is not None
            else thresholds.get("accept", 0.52)
        )
        self.weak_accept_threshold = (
            weak_accept_threshold
            if weak_accept_threshold is not None
            else thresholds.get("weak_accept", 0.45)
        )
        self.min_margin = (
            min_margin
            if min_margin is not None
            else thresholds.get("min_margin", 0.07)
        )

        # Extract intent prototypes from config
        intent_config = config.get("intents", {})
        self.intent_prototypes: dict[str, list[str]] = {
            intent_name: intent_data.get("examples", [])
            for intent_name, intent_data in intent_config.items()
        }

        # Build centroids at init time
        self.intent_centroids = self._build_intent_centroids()

    # ── Public API ────────────────────────────────

    def classify(self, text: str) -> EmbeddingRouterDecision:
        """
        Classify a user input into one of the configured intents.

        Args:
            text: User input text to classify.

        Returns:
            EmbeddingRouterDecision with predicted intent, confidence scores,
            and full breakdown.
        """
        start = time.perf_counter()
        query_embedding = EmbeddingProvider.embed([text])[0]

        raw_scores = {
            intent: float(
                cosine_similarity([query_embedding], [centroid])[0][0]
            )
            for intent, centroid in self.intent_centroids.items()
        }
        latency_s = time.perf_counter() - start

        ranked = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
        top_intent, top_score = ranked[0]
        second_intent, second_score = (
            ranked[1] if len(ranked) > 1 else ("unknown", 0.0)
        )
        margin = top_score - second_score

        accepted = self._is_accepted(top_score=top_score, margin=margin)
        intent = top_intent if accepted else "unknown"

        return EmbeddingRouterDecision(
            input_text=text,
            intent=intent,
            predicted_intent=top_intent,
            second_intent=second_intent,
            confidence=round(top_score, 4),
            second_score=round(second_score, 4),
            margin=round(margin, 4),
            accepted=accepted,
            latency_s=round(latency_s, 4),
            scores={
                intent_name: round(score, 4)
                for intent_name, score in ranked
            },
            model_name=self.model_name,
            embedding=query_embedding.astype(float).tolist(),
        )

    # ── Config Loading ────────────────────────────

    def _load_config(self, config_path: Path) -> dict:
        """Load and validate the YAML configuration file."""
        if not config_path.exists():
            raise FileNotFoundError(
                f"Intent config not found: {config_path}\n"
                f"Please ensure the config file exists or provide a valid path."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config or "intents" not in config:
            raise ValueError(
                f"Invalid config file: {config_path}\n"
                f"Config must contain an 'intents' section with intent definitions."
            )

        return config

    # ── Centroid Construction ─────────────────────

    def _build_intent_centroids(self) -> dict[str, np.ndarray]:
        """
        Compute normalized centroid embeddings for each intent.

        Each centroid is the mean of the embeddings of all prototype examples,
        normalized to unit length for reliable cosine similarity comparisons.
        """
        centroids: dict[str, np.ndarray] = {}
        for intent, examples in self.intent_prototypes.items():
            if not examples:
                continue
            embeddings = EmbeddingProvider.embed(examples)
            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[intent] = centroid
        return centroids

    # ── Acceptance Logic ──────────────────────────

    def _is_accepted(self, top_score: float, margin: float) -> bool:
        """
        Determine whether the top intent prediction is accepted.

        Two-stage acceptance:
        1. Strong accept: confidence >= accept_threshold
        2. Weak accept: confidence >= weak_accept_threshold
           AND margin >= min_margin

        This prevents false positives when the top intent is only slightly
        more similar than the second-best intent.
        """
        if top_score >= self.accept_threshold:
            return True
        return (
            top_score >= self.weak_accept_threshold
            and margin >= self.min_margin
        )