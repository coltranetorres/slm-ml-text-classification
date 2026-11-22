# CLAUDE.md

## Purpose
Mini-research sprint: **when does a small language model (SLM) beat classical ML for support-ticket
classification?** Compares TF-IDF+LogReg, TF-IDF+SVM, MiniLM embeddings+XGBoost, and zero-shot
`gemma-3-1b-it` on accuracy, macro F1, latency and memory, across training-set sizes and label
granularities. Spec: `docs/superpowers/specs/`. Plan: `docs/superpowers/plans/`.

## Commands (uv only — never pip/venv/poetry)
- `uv sync` — install from `uv.lock` (Python 3.12 pinned in `.python-version`)
- `uv add <pkg>` / `uv add --dev <pkg>` — add dependencies (commit `uv.lock`)
- `uv run pytest` — tests (`uv run pytest tests/test_x.py::test_name` for one)
- `uv run ruff check . && uv run ruff format .` — lint + format
- `uv run python scripts/<name>.py` — entry points (`recon_datasets`, `run_sweep`, `make_report`)

## Platform
Apple Silicon (M4 Pro, 24 GB). The SLM runs through **MLX** (`mlx-lm`), which is macOS/arm64 only.
Models are already in `~/.cache/huggingface/hub` (`mlx-community/gemma-3-1b-it-bf16`,
`sentence-transformers/all-MiniLM-L6-v2`). **Ask before downloading any other model or large dataset.**

## Layout
- `src/slmbench/` — `data` (load, splits, subsample), `metrics`, `models` (classical),
  `embeddings` (cached MiniLM), `zeroshot` (Gemma), `bench` (latency/memory), `sweep`, `report`
- `scripts/` — thin CLI entry points only; logic lives in `src/`
- `tests/` — unit tests; `data/` and `results/` are gitignored (except `results/summary.csv`,
  `results/crossover.png`)
- `docs/` — spec, plan, `dataset-choice.md`

## Experiment rules
- Fixed seeds everywhere; splits are stratified and deterministic.
- The **test set is used only for final numbers**. Tune hyperparameters on val.
- Never edit raw data; derived data goes in `data/` and is regenerable.
- Cache embeddings in `data/cache/`.
- Benchmark latency/memory **in an isolated subprocess per model**, batch size 1 after warm-up,
  same test tickets for every model.
- Invalid or unparseable SLM output counts as wrong; never drop rows.
- Report results as measured, including unflattering ones.

## Conventions
- Type hints on public functions; small pure functions; no `print` in `src/` (use `logging`).
- Ruff is the style authority (line length 100).
- TDD: write the failing test first. Unit tests must not hit the network or load real models —
  use stubs/fake embedders; real-model checks live in `scripts/`.
- Keep commits small and conventional (`feat:`, `fix:`, `docs:`, `chore:`).

## Don't
- Commit `data/`, model weights, `.env`, or credentials.
- Use Bitext (templated; makes the comparison trivially easy) as the dataset.
- Fine-tune the SLM — zero-shot only.
