"""Local LLM scoring with MLX (Apple Silicon, free, no API).

Every decision is a multiple-choice letter. Instead of sampling text, one forward pass reads the
model's probability for each allowed letter at the answer position, giving a confidence for every
option in a single call.

Speed: the part of the prompt shared by every call (system prompt, account menu, answer format)
is run through the model once; its key/value cache is kept and trimmed back after each call, so
each call only processes the per-transaction evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

AGENT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
JUDGE_MODELS = {
    "qwen": "mlx-community/Qwen2.5-3B-Instruct-4bit",  # same family as the agent (self-preference check)
    "llama": "mlx-community/Llama-3.2-3B-Instruct-4bit",  # different family
}
ANSWER_PREFIX = "Answer: "


class LetterScorer:
    def __init__(self, model_id: str):
        import mlx.core as mx
        from mlx_lm import load

        # MLX keeps freed GPU buffers for reuse; with a different prompt length on every call that
        # cache grows until an 8 GB machine swaps. Cap it.
        mx.set_cache_limit(256 * 1024**2)
        self.model_id = model_id
        loaded = load(model_id)
        self.model, self.tok = loaded[0], loaded[1]
        self._letter_ids: dict[str, int] = {}
        self._prefix: tuple[str, str] | None = None
        self._prefix_tokens: list[int] = []
        self._cache: Any = None

    def letter_id(self, letter: str) -> int:
        if letter not in self._letter_ids:
            ids = self.tok.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{letter!r} is not a single token for {self.model_id}: {ids}")
            self._letter_ids[letter] = ids[0]
        return self._letter_ids[letter]

    def tokens(self, system: str, user: str) -> list[int]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + ANSWER_PREFIX
        return self.tok.encode(text, add_special_tokens=False)

    def _ensure_prefix(self, system: str, prefix: str) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        if self._prefix == (system, prefix):
            return
        # Shared prefix = longest common token prefix of two prompts that differ only after it.
        a, b = self.tokens(system, prefix + "x"), self.tokens(system, prefix + "q")
        n = next(i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)
        self._prefix_tokens = a[:n]
        self._cache = make_prompt_cache(self.model)
        self.model(mx.array(self._prefix_tokens)[None], cache=self._cache)
        mx.eval([c.state for c in self._cache])
        self._prefix = (system, prefix)

    def _last_logits(self, x):  # noqa: ANN001, ANN202 (mlx arrays)
        """Next-token logits at the last position only. Projecting every position onto a 128k-token
        vocabulary and keeping one row wastes memory; this applies the output head to one row."""
        m = self.model
        inner = getattr(m, "model", None)
        args = getattr(m, "args", None)
        if inner is None or args is None:
            return m(x, cache=self._cache)[0, -1]
        h = inner(x, cache=self._cache)[:, -1:, :]
        head = inner.embed_tokens.as_linear if getattr(args, "tie_word_embeddings", False) else m.lm_head
        return head(h)[0, -1]

    def letter_probs(self, system: str, prefix: str, suffix: str, options: list[str]) -> np.ndarray:
        """Probability of each option letter as the next token, renormalised over the options."""
        import mlx.core as mx
        from mlx_lm.models.cache import trim_prompt_cache

        self._ensure_prefix(system, prefix)
        toks = self.tokens(system, prefix + suffix)
        if toks[: len(self._prefix_tokens)] != self._prefix_tokens:  # tokenizer merged across the seam
            raise RuntimeError("prompt does not start with the cached prefix")
        rest = toks[len(self._prefix_tokens) :]
        logits = self._last_logits(mx.array(rest)[None])
        ids = [self.letter_id(o) for o in options]
        sel = np.array(logits[mx.array(ids)].astype(mx.float32))
        trim_prompt_cache(self._cache, len(rest))
        sel = sel - sel.max()
        p = np.exp(sel)
        return p / p.sum()
