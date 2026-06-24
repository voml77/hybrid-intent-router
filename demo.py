#!/usr/bin/env python3
"""
HybridIntentRouter – Interactive Demo

Shows how the hybrid router classifies user inputs using:
  1. Fast-Path (smalltalk heuristics)
  2. Correction Cache (SQLite)
  3. Local Embedding Router (SentenceTransformer)
  4. Cloud Fallback (simulated for demo)

Run:
    python demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hybrid_router import HybridRouter

def print_separator(char="─", width=72):
    print(char * width)


def print_decision(label, decision):
    print(f"  Input:      {decision.input_text}")
    print(f"  Intent:     {decision.intent}  "
          f"(confidence: {decision.confidence:.2%})")
    print(f"  Source:     {decision.source}")
    print(f"  Accepted:   {decision.accepted}")
    print(f"  Cache Hit:  {decision.cache_hit}")
    print(f"  Fallback:   {decision.fallback_used}")
    print(f"  Latency:    {decision.latency_s:.3f}s")
    print(f"  Top scores:")
    if decision.embedding_decision:
        for intent, score in sorted(
            decision.embedding_decision.scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]:
            bar = "█" * int(score * 40)
            print(f"    {intent:20s}  {score:.2%}  {bar}")
    print()


def main():
    print()
    print("  ⚡ HybridIntentRouter – Demo")
    print("  Intent classification: local embeddings + cloud fallback")
    print()

    router = HybridRouter()

    test_cases = [
        ("💬 Smalltalk",   "Hallo, wie geht es dir heute?"),
        ("📋 Project",     "Wo stehen wir mit dem Projekt? Fasse den Stand zusammen."),
        ("🔍 Code Review", "Kannst du mir bitte diese Python-Datei reviewen?"),
        ("🐛 Debugging",   "Ich bekomme einen Traceback, hilf mir beim Debuggen."),
        ("📁 File Op",     "Bitte lege eine neue Konfigurationsdatei an."),
        ("🧠 Memory",      "Merke dir diese Einstellung für später."),
        ("🌐 Online",      "Recherchiere online nach der aktuellen Version."),
        ("🔒 Privacy",     "Ist das datenschutzkonform? Prüfe die Sicherheit."),
        ("🏗️ Architektur", "Wie sollte die Systemarchitektur aufgebaut sein?"),
        # Zweiter Durchlauf: Testet den Cache!
        ("♻️ Cache Test",  "Hallo, wie geht es dir heute?"),
    ]

    for label, text in test_cases:
        print_separator()
        print(f"  {label}")
        print_separator()
        decision = router.classify(text)
        print_decision(label, decision)

    print_separator("=")
    print("  ✅ Demo complete – hybrid routing pipeline works.")
    print()


if __name__ == "__main__":
    main()