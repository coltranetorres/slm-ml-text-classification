# SLM vs classical ML for support-ticket classification

**Question:** how does a zero-shot small language model (SLM) compare with classical ML classifiers on
support-ticket classification — in accuracy, macro F1, latency and memory — as training data and label
granularity vary, and is an SLM necessary for simple classification at all?

**Short answer:** mostly no, but not never. A zero-shot SLM is the wrong tool when you have even a few hundred
labeled tickets and the labels are simple. It becomes competitive only in a low-label regime on harder
(10-class) labels, and the bigger SLM (4B) shows that scale helps. Every option scores poorly on the hard labels.

![Macro F1 vs training size](results/crossover.png)

## Setup
- **Data:** `Tobi-Bueck/customer-support-tickets`, English only, **exact-duplicate texts removed**
  (23,748 unique tickets; see [docs/dataset-choice.md](docs/dataset-choice.md)). Split 70/15/15, stratified, fixed seed:
  16,622 train / 3,563 val / **3,563 test**. All models are scored on the same test set.
- **Two label granularities:** *coarse* = ticket type (4 classes: Incident, Request, Problem, Change);
  *fine* = support queue (10 classes). These are different label spaces, not a hierarchy: "coarse" means fewer classes.
- **Classical models:** TF-IDF (1–2-grams) + LogReg; TF-IDF + linear SVM (C tuned on val from {0.1, 1, 10} for both);
  `all-MiniLM-L6-v2` embeddings + XGBoost (300 trees, depth 6, lr 0.1; not tuned).
- **Zero-shot SLMs** (same prompt: list the labels, ask for the category name only):
  - **Gemma-3-1B** (`gemma-3-1b-it` bf16), run locally on an M4 Pro via MLX. The label is the one whose first
    distinguishing token has the highest log-probability, so the output is always a valid label.
  - **Gemma-3-4B** (`google/gemma-3-4b-it`) through **OpenRouter** (temperature 0). The reply text is parsed to a
    label; anything unparseable counts as wrong (4 of 3,563 on coarse, 0 on fine).
- **Sweep:** 50 / 200 / 1,000 / full (16,622) training tickets; 3 seeds for 50 and 200 (mean ± std), 1 seed above.
  Zero-shot models need no training, so they are flat lines.
- **Costs:** per-ticket latency at batch size 1 after warm-up; peak RSS measured in an isolated subprocess per local model.

## Results (macro F1 on the test set; full table with accuracy in `results/summary.csv`)

| Model | coarse 50 | coarse 200 | coarse 1k | coarse full | fine 50 | fine 200 | fine 1k | fine full |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.503 | 0.622 | 0.734 | 0.875 | 0.084 | 0.151 | 0.235 | 0.576 |
| TF-IDF + SVM | 0.584 | 0.685 | 0.750 | **0.890** | 0.126 | 0.184 | **0.271** | **0.680** |
| MiniLM + XGBoost | 0.484 | 0.644 | 0.704 | 0.845 | 0.138 | 0.188 | 0.256 | 0.507 |
| Gemma-3-1B zero-shot | 0.366 | 0.366 | 0.366 | 0.366 | 0.133 | 0.133 | 0.133 | 0.133 |
| Gemma-3-4B zero-shot (API) | 0.466 | 0.466 | 0.466 | 0.466 | **0.208** | 0.208 | 0.208 | 0.208 |

Accuracy matters here too, because macro F1 can hide a model that mostly guesses one class. Always predicting the most
common class scores accuracy 0.401 / macro F1 0.143 on coarse and 0.289 / 0.045 on fine.

| Accuracy | coarse | fine |
|---|---|---|
| Majority-class baseline | 0.401 | 0.289 |
| Gemma-3-1B zero-shot | 0.390 | 0.146 |
| Gemma-3-4B zero-shot (API) | 0.654 | 0.288 |
| Classical, 50 tickets | 0.57–0.66 | 0.26–0.28 |
| Classical, full | 0.86–0.88 | 0.58–0.67 |

| Cost | p50 latency / ticket | throughput | peak RSS |
|---|---|---|---|
| TF-IDF + LogReg / SVM | 0.2 ms | ~31–40k tickets/s | 0.22–0.65 GB |
| MiniLM + XGBoost | 3.8–4.6 ms | ~840–1,100 tickets/s | 0.74–1.10 GB |
| Gemma-3-1B (local) | 53 ms (coarse), 75 ms (fine) | 12–16 tickets/s | 3.9 GB |
| Gemma-3-4B (API) | 420–530 ms, includes network | not measured | not applicable (remote) |

RSS for the classical models includes the training process (fit + predict); for Gemma-1B it includes the model weights.
The API latency is a network round trip plus remote inference, so it is not comparable with the local numbers.

## What this says
- **Simple labels (4-class type): the SLM is not needed.** With 50 labeled tickets the SVM (0.58) and LogReg (0.50)
  already beat Gemma-1B (0.37). Gemma-4B (0.47) is roughly equal to what XGBoost gets from ~50 labels, and every
  classical model passes it by 200 tickets (0.62–0.69).
- **Harder labels (10-class queue): there is a real low-label regime for a bigger SLM.** Gemma-4B's 0.208 beats all
  three classical models at 50 and 200 tickets (0.08–0.19). Between 200 and 1,000 tickets the classical models pass it
  (0.24–0.27), and with the full set they are 2.4–3.3× better (0.51–0.68). Gemma-1B (0.133) only competes below
  ~200 tickets, and the classical models are ahead of it from 200 up.
- **Scale helps.** Going from 1B to 4B adds +0.10 macro F1 on coarse and +0.075 on fine, and +0.26 / +0.14 accuracy.
  Gemma-1B's accuracy is below the majority-class baseline on both granularities, so its macro F1 mostly reflects
  spreading predictions across classes, not being right more often.
- **Everyone is weak on the fine labels.** At 50–1,000 tickets the classical models' accuracy (0.26–0.38) is about the
  majority-class level, and the SLMs are not much better (0.288 for 4B). The queue labels are ambiguous in this dataset.
- **Cost:** TF-IDF models are ~250–350× faster per ticket than Gemma-1B and use 6–18× less memory; XGBoost is 12–20×
  faster. The 4B API call is about 2,000× slower per ticket than TF-IDF and sends every ticket to a third party. The
  API cost for this experiment was under $0.10 at $0.04 / $0.05 per 1M tokens.
- **Is an SLM necessary?** For a labeled-data-rich, simple task, no. A zero-shot SLM earns its place when you have almost
  no labels or are cold-starting a harder taxonomy, and then a larger one (4B) is much better than 1B.

## Caveats (read before quoting numbers)
- **The two SLMs are not scored the same way.** Gemma-1B uses label-token log-probabilities locally; Gemma-4B uses
  generated text from a hosted API (unknown provider precision/quantization). Some of the 1B→4B gap could come from
  the method, not only the model size. One prompt, no tuning, no few-shot examples.
- **Dataset:** LLM-generated and multilingual (English only here), with ambiguous queue labels. Results may not
  transfer to real helpdesk data.
- **Deduplication:** we removed exact duplicate texts before splitting (an earlier run without this had ~23% of test
  tickets duplicated in train and overstated full-data classical scores by ~0.08–0.16 macro F1 on fine labels). Near-duplicates
  may remain.
- **Statistics:** one train/val/test split; seed variance exists only at 50 and 200 tickets (3 seeds). The 1k and full
  points have no error bars, and differences of ~0.02 should not be read as real.
- **Ticket truncation is approximate:** 256 tokens locally, 200 words for the API.
- Hardware: Apple M4 Pro, 24 GB. Local latency and memory numbers do not transfer to other machines.

## Reproduce
```bash
uv sync
uv run python scripts/recon_datasets.py        # optional: dataset comparison
uv run python scripts/run_sweep.py             # classical models + Gemma-1B (~20 min on an M4 Pro); resumable
uv run python scripts/run_api_zeroshot.py      # Gemma-4B via OpenRouter; needs OPENROUTER_API_KEY in .env (git-ignored)
uv run python scripts/make_report.py           # results/summary.csv, baselines.csv, crossover.png
uv run pytest
```
Local models used: `mlx-community/gemma-3-1b-it-bf16` and `sentence-transformers/all-MiniLM-L6-v2` (downloaded from
Hugging Face on first use). Project conventions are in [CLAUDE.md](CLAUDE.md).
