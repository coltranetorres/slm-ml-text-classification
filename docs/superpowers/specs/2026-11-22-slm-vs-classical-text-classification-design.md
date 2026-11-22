# SLM vs Classical ML for Support-Ticket Classification — Design

Date: 2026-11-22 · Status: approved · Size: mini-research sprint

## Intent
Answer: **How does a zero-shot SLM compare with classical ML classifiers on support-ticket classification (accuracy, macro F1, latency, memory) as training data and label granularity vary — and is an SLM necessary for simple classification at all?** The sweep is what lets us say *where*, if anywhere, the ranking flips. Output is a small, reproducible experiment plus a short results writeup (table + crossover chart). Owner is a solo researcher; speed and a clear answer beat polish.

## Goals
- Compare four models on one real-looking public ticket dataset:
  1. TF-IDF + Logistic Regression
  2. TF-IDF + linear SVM
  3. Sentence embeddings (all-MiniLM-L6-v2) + XGBoost
  4. SLM zero-shot (no training)
- Metrics: accuracy, macro F1, inference latency, memory.
- Answer "when": sweep training-set size (e.g. 50 / 200 / 1k / full) and label granularity (coarse vs fine) for the three trained models; zero-shot SLM is a flat reference per granularity. Report the crossover point, if any.
- Repo is a uv project with a CLAUDE.md that encodes best practices.

## Non-goals
- No fine-tuning of the SLM (zero-shot only, per brief).
- No second dataset (possible follow-up).
- No hyperparameter-search heroics; modest, equal-effort tuning per classical model.
- No serving/deployment code.

## Anti-goals (failure even if it "works")
- A rigged comparison: a templated/synthetic dataset where TF-IDF trivially hits ~100%.
- Latency/memory numbers that aren't comparable (different batch sizes, warm vs cold, unrelated processes in the measurement).
- Test-set leakage or non-reproducible numbers.

## Constraints
- Hardware: Apple M4 Pro, 24 GB. Python 3.9 on system → uv manages the interpreter (pin 3.12).
- Runtime: **uv** only (`uv sync`, `uv run`, `uv add`); commit `uv.lock`.
- Dataset must download without credentials (Hugging Face datasets preferred; Kaggle needs a token and isn't installed).
- Models are already cached locally in `~/.cache/huggingface/hub`; do not re-download them.

## Approaches considered
- **A. One-shot head-to-head** — fastest; shows who wins but not *when*. Rejected: doesn't answer the research question.
- **B. Head-to-head + training-size / granularity sweep** — **chosen by the user.** Answers "when" cheaply, since the SLM is evaluated only once per granularity.
- **C. B on two datasets** — tests generalization but doubles slow SLM inference. Deferred as follow-up.

## Decisions
**Chosen (user):** approach B.

**My calls (override freely):**
- **SLM (local, MLX via `mlx-lm`) — user decision:** a single model, the smallest Gemma: `mlx-community/gemma-3-1b-it-bf16` (cached). No Qwen, 4-bit variant, or 12B.
- **Zero-shot method:** prompt with the label list, then score each candidate label's log-likelihood (or constrain decoding) so outputs are always valid labels; unparseable output counts as wrong, never dropped.
- **Embeddings:** cached `sentence-transformers/all-MiniLM-L6-v2`; embeddings computed once and cached to disk. Embedding time counts toward model 3's latency.
- **Dataset:** selected in the first implementation step by a short recon of 2–3 public HF ticket datasets. Criteria: real (non-templated) text, ≥ ~5k rows, a coarse label with ~5–15 classes and ideally a finer label, class imbalance documented. Bitext is excluded (templated). Fallback: banking77 with intents grouped into coarse categories. Record the choice and why in the README.
- **Splits:** stratified train/val/test, fixed seed; test set touched only for final numbers. Sweep subsamples are nested and seeded; repeat 3 seeds for the small-N points and report mean ± std.
- **Latency:** per-ticket at batch size 1 (p50/p95) after warm-up, plus batched throughput; same test subset for all models. For TF-IDF models, include vectorization; for model 3, include embedding.
- **Memory:** peak RSS measured in an isolated subprocess per model (includes model load), not the whole experiment process.
- **Layout:** `src/<pkg>/` (data, models, bench, report), `scripts/` entry points, `data/` and `results/` gitignored except small summary CSVs, `tests/` for splitting/metrics/label-scoring logic, ruff + pytest in dev deps.
- **Output:** `results/summary.csv`, a crossover chart (macro F1 vs training size, SLM as horizontal line), and a short `README` writeup answering the research question.

**CLAUDE.md contents:** project purpose; uv commands; Apple-silicon/MLX note; repo layout; experiment rules (seeds, no test-set tuning, isolated benchmarking, never edit raw data, cache embeddings); coding conventions (type hints, ruff, small pure functions); testing expectations; "ask before downloading large models or datasets"; where results go.

## Left to the builder
Exact prompt wording, XGBoost/SVM/LR hyperparameters (small grid on val), training sizes if the dataset is small, chart styling, and the final dataset pick within the criteria above.

## Open risks
- Zero-shot 1–1.5B models may do poorly on fine-grained labels; that is a valid finding, not a bug.
- If the chosen dataset has noisy labels, all models cap out low and the comparison flattens; recon should check this with a quick TF-IDF baseline.
