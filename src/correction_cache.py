from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path


# ── Default: Local DB in the project directory ───
DEFAULT_DB_PATH = Path.home() / ".hybrid_intent_router" / "correction_cache.db"


class CorrectionCache:
    """
    SQLite-based exact-match correction cache for intent routing.

    Stores previously resolved intent classifications so that
    recurring inputs can bypass local embedding and cloud fallback.

    The cache learns from cloud-fallback corrections, reducing
    latency and cost over time.

    Schema:
        input_hash: SHA-256 of normalized input (unique, primary key)
        normalized_input: Lower-cased, whitespace-normalized text
        final_intent: The resolved intent
        local_prediction: What the local EmbeddingRouter predicted
        local_confidence: Confidence of the local prediction
        fallback_intent: What the cloud fallback predicted
        fallback_confidence: Confidence of the fallback
        hit_count: How often this entry was looked up
        last_seen: Timestamp of last lookup
        created_at: Timestamp of first creation
        updated_at: Timestamp of last update
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._ensure_db()

    # ── Public API ────────────────────────────────

    def get_intent(self, text: str) -> str | None:
        """
        Look up an input by exact normalized hash.

        Args:
            text: Raw user input.

        Returns:
            The resolved intent if found, else None.
        """
        input_hash = self._build_hash(text)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT final_intent FROM correction_cache WHERE input_hash = ?",
                (input_hash,),
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def register_hit(self, text: str) -> None:
        """
        Increment hit count and update last_seen for a cached input.

        Args:
            text: Raw user input.
        """
        input_hash = self._build_hash(text)
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE correction_cache
                SET hit_count = hit_count + 1, last_seen = ?, updated_at = ?
                WHERE input_hash = ?
                """,
                (now, now, input_hash),
            )
            conn.commit()

    def save_correction(
        self,
        text: str,
        final_intent: str,
        local_prediction: str | None = None,
        local_confidence: float | None = None,
        fallback_intent: str | None = None,
        fallback_confidence: float | None = None,
    ) -> None:
        """
        Save or update a correction entry.

        Args:
            text: Raw user input.
            final_intent: The resolved intent.
            local_prediction: What the local router predicted (optional).
            local_confidence: Confidence of the local prediction (optional).
            fallback_intent: What the fallback predicted (optional).
            fallback_confidence: Confidence of the fallback (optional).
        """
        normalized = self._normalize(text)
        input_hash = self._build_hash(text)
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO correction_cache (
                    input_hash, normalized_input, final_intent,
                    local_prediction, local_confidence,
                    fallback_intent, fallback_confidence,
                    hit_count, last_seen, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                ON CONFLICT(input_hash) DO UPDATE SET
                    final_intent = excluded.final_intent,
                    local_prediction = excluded.local_prediction,
                    local_confidence = excluded.local_confidence,
                    fallback_intent = excluded.fallback_intent,
                    fallback_confidence = excluded.fallback_confidence,
                    updated_at = excluded.updated_at,
                    last_seen = excluded.last_seen
                """,
                (
                    input_hash, normalized, final_intent,
                    local_prediction, local_confidence,
                    fallback_intent, fallback_confidence,
                    now, now, now,
                ),
            )
            conn.commit()

    # ── Internals ─────────────────────────────────

    def _normalize(self, text: str) -> str:
        """Normalize input: lowercase, collapse whitespace."""
        return " ".join(text.strip().lower().split())

    def _build_hash(self, text: str) -> str:
        """Build a SHA-256 hash of the normalized input."""
        normalized = self._normalize(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _ensure_db(self) -> None:
        """Create the database and table if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS correction_cache (
                    input_hash TEXT PRIMARY KEY,
                    normalized_input TEXT NOT NULL,
                    final_intent TEXT NOT NULL,
                    local_prediction TEXT,
                    local_confidence REAL,
                    fallback_intent TEXT,
                    fallback_confidence REAL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    last_seen TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
