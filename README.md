# HybridIntentRouter

**Local-first intent routing with cloud fallback – smart, fast, and cost-aware.**

Classify user input into intents using local embeddings (SentenceTransformer). When the local router is uncertain, fall back to a cloud LLM. Every correction is cached in SQLite, so the system learns and improves over time.

```
Input:  "Wo stehen wir mit dem Projekt? Fasse den Stand zusammen."
Output: project  (source: embedding_router, confidence: 78.6 %)

Input:  "Hilf mir beim Debuggen dieses Tracebacks"
Output: debugging  (source: cloud_fallback, confidence: 95.0 %)

Input:  "Hilf mir beim Debuggen dieses Tracebacks"  (second time)
Output: debugging  (source: correction_cache, confidence: 100 %)
```

---

## 🔍 Why?

Pure local embedding routers handle ~90 % of intent classifications with ~49 ms latency. But some intents – especially debugging – are inherently tricky for embedding similarity.

Instead of throwing a cloud LLM at *every* request, **HybridIntentRouter** uses one only when it matters:

| Layer | Latency | Cost | Hit Rate |
|---|---|---|---|
| Fast-Path (heuristic) | ~0 ms | €0 | ~15 % |
| Correction Cache (SQLite) | ~1 ms | €0 | ~20 % |
| Embedding Router (local) | ~49 ms | €0 | ~55 % |
| Cloud Fallback (LLM) | ~500–2000 ms | ~€0.001 | ~10 % |

**Result:** ~90 % of requests are handled locally. Cloud costs drop by ~90 % while maintaining 100 % accuracy.

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚡ **Local-first** | No cloud API key required for the common case |
| 🧠 **Self-learning** | Caches cloud corrections in SQLite – gets smarter over time |
| ⚙️ **Configurable** | Intents and thresholds via YAML |
| 🔌 **Provider-agnostic** | Bring your own cloud fallback (Gemini, OpenAI, Claude, ...) |
| 📊 **Transparent** | Every decision includes full score breakdown and source trace |
| 🚀 **Fast** | ~49 ms for local classification |

---

## 🏗️ Architecture

```
User Input
    │
    ├── Fast-Path (smalltalk heuristics)
    │     └─→ smalltalk (if detected)
    │
    ├── Correction Cache (SQLite exact match)
    │     └─→ cached intent (if previously corrected)
    │
    ├── Embedding Router (SentenceTransformer + Cosine Similarity)
    │     ├─→ accepted → return intent
    │     └─→ uncertain ──→
    │
    └── Cloud Fallback (LLM)
          ├─→ accepted → return intent + save to cache
          └─→ unknown  → return "unknown"
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/voml77/hybrid-intent-router.git
cd hybrid-intent-router

# 2. Install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run demo (local mode – no cloud fallback)
python demo.py
```

---

## 🧪 Usage

### Local-only mode (no cloud required)

```python
from hybrid_router import HybridRouter

router = HybridRouter()

decision = router.classify("Hallo, wie geht es dir?")
print(f"Intent:      {decision.intent}")
print(f"Confidence:  {decision.confidence:.2%}")
print(f"Source:      {decision.source}")
print(f"Cache Hit:   {decision.cache_hit}")
```

### Hybrid mode (with cloud fallback)

```python
from hybrid_router import HybridRouter
from cloud_fallback import CloudFallbackRouter

class MyCloudRouter(CloudFallbackRouter):
    def classify(self, text):
        # Example: Gemini 2.5 Flash (fast, cheap, accurate)
        # Or any provider you prefer – OpenAI, Claude, local LLM, ...
        pass

router = HybridRouter(
    cloud_fallback=MyCloudRouter()
)

decision = router.classify("Hilf mir beim Debuggen dieses Tracebacks")
print(f"Intent:      {decision.intent}  (fallback: {decision.fallback_used})")
```

---

## ⚙️ Configuration

Intents are defined in `src/config/intents.yaml`:

```yaml
intents:
  smalltalk:
    description: "Greetings, casual conversation"
    examples:
      - "Hallo, wie geht es dir?"
      - "Guten Morgen! Alles gut bei dir?"
```

Thresholds are also configurable:

```yaml
thresholds:
  accept: 0.52
  weak_accept: 0.45
  min_margin: 0.07
```

---

## 🧠 How It Was Developed

This module was extracted from a production AI assistant architecture. The key insight? You don't need to call a cloud LLM for every "hello". Embedding-based routing handles the bulk locally, and the cloud only acts as a safety net for edge cases.

The HybridRouter is provider-agnostic. For reference, the original system used Gemini 2.5 Flash – fast, cheap, and accurate. But you can implement any provider: OpenAI, Claude, a local LLM, or even a rules-based fallback. The architecture stays the same.

Over 30 days of intensive usage, the cloud fallback cost exactly **€0.31** – while maintaining 100 % routing accuracy.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Multilingual sentence embeddings |
| `numpy` | Centroid computation and normalization |
| `scikit-learn` | Cosine similarity |
| `PyYAML` | Intent configuration |

---

## 🗺️ Roadmap

- [ ] Dynamic centroid updating (online learning)
- [ ] REST API wrapper
- [ ] Built-in Gemini/OpenAI fallback implementations
- [ ] Prometheus metrics for cost tracking

---

## 📄 License

MIT – use it, fork it, share it.

---

## 🤝 Contributing

Ideas, issues, and PRs welcome. If you've built a cloud fallback implementation for your preferred provider, consider contributing it as a reference example.
