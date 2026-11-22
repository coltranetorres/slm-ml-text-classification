# Dataset choice

**Chosen: `Tobi-Bueck/customer-support-tickets`, English rows only** (28,261 tickets, mean 410 chars).
No credentials needed. Text = `subject + "\n" + body`.

| Granularity | Column | Classes | TF-IDF+LR macro-F1 (quick 80/20) |
|---|---|---|---|
| coarse | `type` (Incident / Request / Problem / Change) | 4 | 0.877 |
| fine | `queue` (10 support queues, 8149 … 404 per class) | 10 | 0.476 |

Candidates (`scripts/recon_datasets.py`):
- `tasksource/it-support-tickets`: 2.2k rows, 7 classes, one label only, very short text (164 chars). Too small and no second granularity.
- `PolyAI/banking77`: 77 intents, short queries (59 chars). Would need an invented coarse mapping.
- Bitext: excluded (templated).

**Why this one:** real-length multi-sentence tickets, two label granularities, and a classical baseline that is
neither trivial (queue 0.48) nor saturated. Caveat: the corpus is LLM-generated and multilingual
(we keep English only), and **`type` and `queue` are two different label spaces, not a hierarchy**
("coarse" = fewer classes, "fine" = more). Report this honestly in the README.
