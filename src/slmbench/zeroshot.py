"""Zero-shot SLM classifier: pick the label whose tokens the model finds most likely."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from slmbench.metrics import INVALID

MODEL_ID = "mlx-community/gemma-3-1b-it-bf16"


def build_prompt(ticket: str, labels: Sequence[str]) -> str:
    options = "\n".join(f"- {label}" for label in labels)
    return (
        "You are a customer support triage assistant. Classify the support ticket into "
        "exactly one of these categories:\n"
        f"{options}\n\n"
        f"Ticket:\n{ticket}\n\n"
        "Answer with the category name only."
    )


def truncate_text(
    text: str,
    max_tokens: int,
    encode: Callable[[str], list],
    decode: Callable[[list], str],
) -> str:
    toks = encode(text)
    return text if len(toks) <= max_tokens else decode(toks[:max_tokens])


def pick_label(scores: Mapping[str, float]) -> str:
    return max(scores, key=scores.__getitem__) if scores else INVALID


def unique_prefixes(label_tokens: Mapping[str, list[int]]) -> dict[str, list[int]]:
    """Shortest token prefix of each label that no other label shares."""
    out: dict[str, list[int]] = {}
    for label, toks in label_tokens.items():
        for k in range(1, len(toks) + 1):
            prefix = toks[:k]
            clash = any(o != label and other[:k] == prefix for o, other in label_tokens.items())
            if not clash or k == len(toks):
                out[label] = prefix
                break
        else:
            out[label] = toks
    return out


class ZeroShotClassifier:
    name = "gemma-3-1b-zeroshot"

    def __init__(
        self,
        labels: Sequence[str],
        model_id: str = MODEL_ID,
        max_ticket_tokens: int = 256,
    ) -> None:
        from mlx_lm import load

        self.labels = list(labels)
        self.max_ticket_tokens = max_ticket_tokens
        self._model, self._tok = load(model_id)
        label_tokens = {
            label: self._tok.encode(label, add_special_tokens=False) for label in self.labels
        }
        self._prefixes = unique_prefixes(label_tokens)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        """Zero-shot: nothing to learn."""

    def _prompt_ids(self, ticket: str) -> list[int]:
        enc = lambda s: self._tok.encode(s, add_special_tokens=False)  # noqa: E731
        ticket = truncate_text(ticket, self.max_ticket_tokens, enc, self._tok.decode)
        chat = [{"role": "user", "content": build_prompt(ticket, self.labels)}]
        text = self._tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        return enc(text)

    def _scores(self, ticket: str) -> dict[str, float]:
        import mlx.core as mx

        ids = self._prompt_ids(ticket)
        if all(len(p) == 1 for p in self._prefixes.values()):
            logits = self._model(mx.array([ids]))[0, -1].astype(mx.float32)
            logp = logits - mx.logsumexp(logits)
            return {label: float(logp[p[0]]) for label, p in self._prefixes.items()}
        scores = {}
        for label, prefix in self._prefixes.items():
            logits = self._model(mx.array([ids + prefix]))[0].astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            start = len(ids) - 1
            scores[label] = sum(float(logp[start + i, tok]) for i, tok in enumerate(prefix))
        return scores

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [pick_label(self._scores(t)) for t in texts]
