# SLM vs Classical Text Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A uv project that benchmarks TF-IDF+LR, TF-IDF+SVM, MiniLM+XGBoost and zero-shot Gemma-3-1B on support-ticket classification, with a training-size × label-granularity sweep, and writes a results table, crossover chart and short README answer.

**Architecture:** `src/slmbench/` holds small modules (data, metrics, models, zeroshot, bench, sweep, report). All models implement one `Classifier` protocol (`fit`, `predict`) so the sweep and benchmark treat them uniformly. Latency/memory are measured in an isolated subprocess per model. Scripts in `scripts/` are thin entry points.

**Tech Stack:** Python 3.12, uv, scikit-learn, xgboost, sentence-transformers, mlx-lm, datasets, pandas, matplotlib, psutil, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-22-slm-vs-classical-text-classification-design.md`

## Global Constraints

- Runtime is **uv only** (`uv sync`, `uv run`, `uv add`); pin Python 3.12; commit `uv.lock`.
- SLM is exactly `mlx-community/gemma-3-1b-it-bf16` (cached locally); embeddings are `sentence-transformers/all-MiniLM-L6-v2` (cached). Never download other models; ask before any large download.
- Dataset must need no credentials (Hugging Face preferred). Bitext is excluded (templated). Fallback: banking77 with intents grouped to coarse categories.
- Zero-shot only; no SLM fine-tuning. One dataset only.
- Fixed seeds; stratified train/val/test; test set used only for final numbers; hyperparameters tuned on val only.
- Sweep: training sizes 50 / 200 / 1000 / full (clip to dataset), 3 seeds for small sizes, report mean ± std; coarse and fine label granularity.
- Latency: batch size 1 after warm-up, p50/p95, plus batched throughput, same test subset for all models; TF-IDF includes vectorization, embed+XGB includes embedding. Memory: peak RSS in an isolated subprocess including model load.
- Unparseable/invalid SLM output counts as wrong, never dropped.
- `data/` and `results/` gitignored except small summary CSVs/chart.

## Review Focus

- Subsample smaller than class count (n=50 with 20+ classes): stratified sampling must not crash; classes missing from the train subset are still valid test labels (counted wrong, not dropped).
- Empty/whitespace-only ticket text: every model returns a label without raising.
- Very long tickets: SLM prompt truncates ticket text to a fixed token budget instead of overflowing context.
- XGBoost requires 0..K-1 integer labels, and the label set seen in a subsample may be non-contiguous: encode via a fitted `LabelEncoder`.
- Non-English / non-ASCII text does not crash the vectorizers or tokenizer.

---

### Task 1: Project scaffold and CLAUDE.md

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `CLAUDE.md`, `src/slmbench/__init__.py`, `tests/test_smoke.py`
- Create dirs: `scripts/`, `data/`, `results/`

**Interfaces:**
- Produces: importable package `slmbench`; `uv run pytest` and `uv run ruff check .` work.

- [ ] **Step 1: Write failing test** `tests/test_smoke.py::test_package_imports` asserting `import slmbench` works and `slmbench.__version__ == "0.1.0"`.
- [ ] **Step 2: Run** `uv run pytest` — Expected: FAIL (no project yet).
- [ ] **Step 3: Scaffold** with `uv init --package --name slmbench --python 3.12` (src layout), then `uv add scikit-learn xgboost sentence-transformers mlx-lm datasets pandas matplotlib psutil` and `uv add --dev pytest ruff`. Configure ruff (line length 100, rules E,F,I,UP,B) and pytest (`testpaths=["tests"]`) in `pyproject.toml`. `.gitignore`: `.venv`, `data/`, `results/*` except `results/summary.csv` and `results/crossover.png`, caches.
- [ ] **Step 4: Write `CLAUDE.md`** (under ~80 lines) with: project purpose (one paragraph); commands (`uv sync`, `uv run pytest`, `uv run ruff check . && uv run ruff format .`, `uv run python scripts/<name>.py`); Apple-silicon/MLX note; module map; experiment rules (seeds, no test-set tuning, isolated benchmarking, never edit raw data, cache embeddings in `data/cache/`); conventions (type hints, small pure functions, no print in `src/`, use `logging`); testing expectations (unit-test splitting, metrics, label scoring; no network or model loads in unit tests); "ask before downloading large models/datasets"; never commit `data/` or secrets.
- [ ] **Step 5: Run** `uv run pytest && uv run ruff check .` — Expected: PASS.
- [ ] **Step 6: Commit** `chore: scaffold uv project and CLAUDE.md`.

### Task 2: Dataset recon and loader

**Files:**
- Create: `scripts/recon_datasets.py`, `src/slmbench/data.py`, `tests/test_data.py`, `docs/dataset-choice.md`

**Interfaces:**
- Produces:
  - `load_tickets() -> pd.DataFrame` with columns `text: str`, `label_coarse: str`, `label_fine: str`; cached to `data/tickets.parquet`.
  - `@dataclass Splits(train, val, test: pd.DataFrame)`; `make_splits(df, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.15) -> Splits` (stratified on `label_fine`).
  - `subsample(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame` — nested (the n=50 sample ⊂ the n=200 sample for the same seed), stratified where possible, never raises when `n` < class count or `n` ≥ len(train) (returns all).

- [ ] **Step 1: Recon.** Write `scripts/recon_datasets.py` that, for 2–3 candidate HF ticket datasets (candidates found by searching the Hub; exclude Bitext), prints row count, label counts, mean text length, class imbalance, and a quick TF-IDF+LR val macro-F1. Run it.
- [ ] **Step 2: Choose.** Pick by the spec criteria (real text, ≥ ~5k rows, coarse ~5–15 classes + finer label, TF-IDF not trivially ~100% and not noise-capped). Record the choice, candidates' numbers and reasoning in `docs/dataset-choice.md`. **Checkpoint: tell the user the pick before continuing.** If no candidate has two granularities, fall back to banking77 intents → coarse groups and document the mapping.
- [ ] **Step 3: Write failing tests** in `tests/test_data.py` on a small synthetic DataFrame: `test_splits_are_disjoint_and_stratified`, `test_splits_deterministic_for_seed`, `test_subsample_nested` (50-subset ⊂ 200-subset), `test_subsample_smaller_than_class_count_does_not_raise`, `test_subsample_larger_than_train_returns_all`.
- [ ] **Step 4: Run** `uv run pytest tests/test_data.py` — Expected: FAIL.
- [ ] **Step 5: Implement** `load_tickets`, `make_splits`, `subsample` as above. Nesting: shuffle each class with the seed once and take prefixes.
- [ ] **Step 6: Run** tests — Expected: PASS. Commit `feat: dataset loader, splits, nested subsampling`.

### Task 3: Metrics and result schema

**Files:** Create `src/slmbench/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces: `evaluate(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]` returning `accuracy` and `macro_f1`, where macro F1 averages over the union of true and predicted labels so a never-predicted/invalid label (`"__invalid__"`) is penalized, not dropped. `Result = TypedDict` / dataclass `ResultRow(model, granularity, n_train, seed, accuracy, macro_f1, p50_ms, p95_ms, throughput_per_s, peak_rss_mb)`.

- [ ] **Step 1: Write failing tests:** `test_perfect_predictions` (acc 1.0, f1 1.0); `test_invalid_prediction_counts_wrong` (`["a","b"]` vs `["a","__invalid__"]` → acc 0.5, macro_f1 < 1); `test_label_missing_from_predictions_lowers_macro_f1`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** `evaluate` with scikit-learn (`zero_division=0`). **Step 4: Run** — PASS. **Step 5: Commit** `feat: metrics and result schema`.

### Task 4: Classical models

**Files:** Create `src/slmbench/models.py`, `src/slmbench/embeddings.py`, `tests/test_models.py`

**Interfaces:**
- Produces:
  - `class Classifier(Protocol): name: str; def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None; def predict(self, texts: Sequence[str]) -> list[str]`.
  - `build_tfidf_lr() -> Classifier`, `build_tfidf_svm() -> Classifier` (TF-IDF word 1–2-grams, sublinear tf; LR and `LinearSVC`; C tuned on val in the sweep via `tune_C(...)`-style small grid `[0.1, 1, 10]`, passed as argument `C: float = 1.0`).
  - `build_embed_xgb(embedder: Embedder, **xgb_params) -> Classifier`; `Embedder` wraps cached `all-MiniLM-L6-v2` with `embed(texts: Sequence[str]) -> np.ndarray` and a disk cache in `data/cache/` keyed by text hash. Labels encoded with a fitted `LabelEncoder` inside the classifier.

- [ ] **Step 1: Write failing tests** (tiny synthetic corpus, a fake embedder returning deterministic vectors so no model loads): `test_each_model_fits_and_predicts_known_labels`; `test_empty_string_prediction_does_not_raise` (all three); `test_non_ascii_text_ok`; `test_xgb_handles_noncontiguous_label_subset`; `test_embedder_cache_hits_disk` (second call doesn't re-encode, using a stub encode function).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** with scikit-learn pipelines; XGBoost defaults `n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist"`. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat: tfidf-lr, tfidf-svm, embed-xgb classifiers`.

### Task 5: Zero-shot SLM classifier

**Files:** Create `src/slmbench/zeroshot.py`, `tests/test_zeroshot.py`

**Interfaces:**
- Consumes: `Classifier` protocol (Task 4).
- Produces: `class ZeroShotClassifier` with `name = "gemma-3-1b-zeroshot"`, `__init__(labels: Sequence[str], model_id: str = "mlx-community/gemma-3-1b-it-bf16", max_ticket_tokens: int = 256)`; `fit` is a no-op (stores nothing); `predict(texts) -> list[str]`. Pure helpers (unit-testable without the model): `build_prompt(ticket: str, labels: Sequence[str]) -> str` and `pick_label(scores: Mapping[str, float]) -> str`.

- [ ] **Step 1: Write failing tests** for the pure helpers: `test_prompt_contains_all_labels_and_ticket`; `test_prompt_truncates_long_ticket` (ticket of 10k words → prompt token-budget respected using a stub tokenizer); `test_pick_label_argmax`; `test_pick_label_empty_scores_returns_invalid_sentinel` (`"__invalid__"`).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** Scoring: for each ticket, compute the log-likelihood of each candidate label string as the continuation of the prompt (length-normalized), using `mlx_lm` model forward passes; share the prompt prefix KV where simple, otherwise score labels in one batched pass. Chosen label = argmax, so outputs are always valid labels; the sentinel is only for empty scores.
- [ ] **Step 4: Run** unit tests — PASS.
- [ ] **Step 5: Smoke (real model, manual)** — `uv run python scripts/smoke_zeroshot.py` classifies 5 hand-written tickets with 3 labels; Expected: runs, returns valid labels, prints per-ticket seconds. Confirms model loads offline from the cache.
- [ ] **Step 6: Commit** `feat: zero-shot gemma classifier via label log-likelihood`.

### Task 6: Benchmark harness (latency + memory)

**Files:** Create `src/slmbench/bench.py`, `scripts/bench_worker.py`, `tests/test_bench.py`

**Interfaces:**
- Consumes: `Classifier`, `evaluate`.
- Produces: `summarize_latencies(ms: Sequence[float]) -> tuple[float, float]` (p50, p95); `run_isolated(model_spec: dict, train_path: str, test_path: str, warmup: int = 5, n_latency: int = 200) -> dict` which launches `bench_worker.py` as a subprocess (`sys.executable`), and returns `accuracy, macro_f1, p50_ms, p95_ms, throughput_per_s, peak_rss_mb` parsed from the worker's JSON stdout. Worker records peak RSS with `psutil`/`resource.getrusage` after load + predict.

- [ ] **Step 1: Write failing tests:** `test_summarize_latencies_known_values`; `test_run_isolated_returns_all_keys` using a trivial model spec (`tfidf_lr` on a 20-row parquet fixture, `n_latency=5`).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** Worker: build model from spec, fit (if trainable), warm up, time `predict([t])` for `n_latency` test tickets at batch size 1, then time one batched predict over the same tickets for throughput, evaluate on the full test set. **Step 4: Run** — PASS. **Step 5: Commit** `feat: isolated latency/memory benchmark`.

### Task 7: Sweep runner

**Files:** Create `src/slmbench/sweep.py`, `scripts/run_sweep.py`, `tests/test_sweep.py`

**Interfaces:**
- Consumes: `load_tickets`, `make_splits`, `subsample`, model builders, `run_isolated`.
- Produces: `run_sweep(sizes: Sequence[int], seeds: Sequence[int], granularities: Sequence[str], out_csv: Path) -> pd.DataFrame` — appends/writes `ResultRow`s; resumable (skips rows already in the CSV). Zero-shot is run once per granularity (no sizes/seeds). `C` for LR/SVM picked on val from `[0.1, 1, 10]` per (size, seed). `aggregate(df) -> pd.DataFrame` gives mean ± std per (model, granularity, n_train).

- [ ] **Step 1: Write failing tests** with a stubbed `run_isolated` and tiny dataset: `test_sweep_writes_expected_row_count`; `test_sweep_is_resumable` (second call adds no rows); `test_zeroshot_runs_once_per_granularity`; `test_aggregate_mean_std`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Real run:** `uv run python scripts/run_sweep.py` (sizes 50 200 1000 full; seeds 0 1 2 for sizes ≤ 200, seed 0 otherwise; both granularities). Expected: `results/raw.csv` complete; log elapsed time. If the SLM pass on the full test set is too slow, cap the test set for **all** models at a documented fixed size and say so in the README.
- [ ] **Step 6: Commit** `feat: sweep runner`.

### Task 8: Report, chart and README

**Files:** Create `src/slmbench/report.py`, `scripts/make_report.py`, `README.md`, `tests/test_report.py`

**Interfaces:**
- Produces: `make_summary(raw: pd.DataFrame) -> pd.DataFrame` → `results/summary.csv`; `plot_crossover(summary: pd.DataFrame, out: Path) -> None` — one panel per granularity, macro F1 vs training size (log x) with error bars for trained models and a horizontal line for zero-shot Gemma; second small figure/table for latency and peak RSS per model.

- [ ] **Step 1: Write failing tests:** `test_summary_has_one_row_per_model_granularity_size`; `test_plot_crossover_writes_png` (tmp_path).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** (load the `dataviz` skill before writing chart code). **Step 4: Run** — PASS.
- [ ] **Step 5: Generate** `uv run python scripts/make_report.py`; open the PNG and check it reads correctly.
- [ ] **Step 6: Write `README.md`:** research question, dataset (link to `docs/dataset-choice.md`), method (one paragraph), results table, chart, and a direct answer: where (if anywhere) the classical models overtake zero-shot Gemma, cost in latency/memory, and whether an SLM is necessary here. Report numbers as measured, including unflattering ones.
- [ ] **Step 7: Run** `uv run pytest && uv run ruff check .` — Expected: all pass. Commit `docs: results, chart and README`.

---

## Self-review

- **Spec coverage:** four models (T4, T5); metrics acc/macro-F1 (T3), latency/memory isolated (T6); sweep over size and granularity with seeds (T7); dataset recon with criteria and fallback (T2); CLAUDE.md, uv, layout, ruff/pytest (T1); chart + README answer (T8); single smallest Gemma (Global Constraints, T5). Optional 4-bit/Qwen/12B dropped per user decision.
- **Types:** `Classifier`, `ResultRow`, `Splits`, `evaluate`, `subsample`, `run_isolated` names are consistent across tasks.
- **Review Focus** each has a test: small subsample (T2), empty text (T4), long tickets (T5), XGB labels (T4), non-ASCII (T4); invalid SLM output penalty (T3).
