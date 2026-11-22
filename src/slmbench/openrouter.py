"""Zero-shot classification through the OpenRouter chat API (remote SLM, e.g. gemma-3-4b-it).

Latency here is network round-trip + remote inference; memory is not measurable locally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from slmbench.metrics import INVALID
from slmbench.zeroshot import build_prompt, truncate_text

log = logging.getLogger(__name__)

MODEL = "google/gemma-3-4b-it"
BASE_URL = "https://openrouter.ai/api/v1"
CACHE_DIR = Path("data/cache")
MAX_TICKET_WORDS = 200  # rough stand-in for the local 256-token budget


def parse_label(reply: str, labels: Sequence[str]) -> str:
    """Map a free-text reply to one label, or INVALID if empty/unknown/ambiguous."""
    clean = reply.strip().strip("\"'`*.:;,! \n\t").lower()
    for label in labels:
        if clean == label.lower():
            return label
    low = reply.lower()
    hits = [label for label in labels if label.lower() in low]
    hits = [h for h in hits if not any(h != o and h.lower() in o.lower() for o in hits)]
    return hits[0] if len(hits) == 1 else INVALID


def make_complete(model: str = MODEL) -> Callable[[str], str]:
    """Real OpenRouter call; reads OPENROUTER_API_KEY from the environment / .env."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])

    def complete(prompt: str) -> str:
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        return out.choices[0].message.content or ""

    return complete


class OpenRouterClassifier:
    def __init__(
        self,
        labels: Sequence[str],
        model: str = MODEL,
        name: str = "gemma-3-4b-api-zeroshot",
        complete: Callable[[str], str] | None = None,
        cache_dir: Path = CACHE_DIR,
        workers: int = 16,
        max_retries: int = 6,
        backoff_s: float = 1.0,
    ) -> None:
        self.name, self.labels, self.model = name, list(labels), model
        self._complete = complete
        self.workers, self.max_retries, self.backoff_s = workers, max_retries, backoff_s
        self._lock = threading.Lock()
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = cache_dir / f"openrouter_{model.replace('/', '_')}.jsonl"
        self._cache: dict[str, str] = {}
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                rec = json.loads(line)
                self._cache[rec["k"]] = rec["v"]

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        """Zero-shot: nothing to learn."""

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{prompt}".encode()).hexdigest()

    def _call(self, prompt: str) -> str:
        if self._complete is None:
            self._complete = make_complete(self.model)
        for attempt in range(self.max_retries):
            try:
                return self._complete(prompt)
            except Exception:
                if attempt == self.max_retries - 1:
                    raise  # never silently score an API failure as a wrong label
                time.sleep(self.backoff_s * 2**attempt)
        raise AssertionError("unreachable")

    def _reply(self, ticket: str, use_cache: bool = True) -> str:
        ticket = truncate_text(ticket, MAX_TICKET_WORDS, str.split, " ".join)
        prompt = build_prompt(ticket, self.labels)
        key = self._key(prompt)
        if use_cache and key in self._cache:
            return self._cache[key]
        reply = self._call(prompt)
        if use_cache:
            with self._lock:
                self._cache[key] = reply
                with self._path.open("a") as f:
                    f.write(json.dumps({"k": key, "v": reply}) + "\n")
        return reply

    def predict_one(self, ticket: str, use_cache: bool = True) -> str:
        return parse_label(self._reply(ticket, use_cache), self.labels)

    def predict(self, texts: Sequence[str]) -> list[str]:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.predict_one, texts))
