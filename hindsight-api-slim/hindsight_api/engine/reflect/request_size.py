"""Size a reflect request in the provider's tokens before it is sent.

``HINDSIGHT_API_REFLECT_MAX_CONTEXT_TOKENS`` is meant to bound what the provider
bills, but the agent loop counted only the message text with the configured local
encoding. That count misses the tool definitions sent with every turn and the gap
between the local encoding and the provider's own tokenizer (>= 18% measured for
Claude against o200k_base), so a 100K budget let requests of ~120K go out.

Every tool-calling turn reports the real prompt size it was billed for. The sizer
learns the ratio between that report and the local count of the same request
(messages plus tool definitions) and scales the next request's local count by the
largest ratio seen in this run. Until a provider reports usage -- the first turn,
or a provider that reports none -- the estimate is the plain local message count,
exactly as before, so nothing changes where nothing is known.
"""

import json
import math
from typing import Any

from .tokenization import count_prompt_tokens

#: Upper bound on the learned ratio. Real tokenizer gaps are well under 2x; a ratio
#: above this means the provider reported something other than this one request
#: (e.g. usage summed over retries), and trusting it would end every reflect early.
_MAX_TOKENIZER_RATIO = 3.0


class RequestSizer:
    """Estimates the provider-token size of the next reflect request."""

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        # Counted once per run, not at import: importing reflect must not load the
        # tokenizer (test_reflect_tokenizer_lazy_load).
        self._tool_tokens = count_prompt_tokens(json.dumps(tools, ensure_ascii=False, default=str))
        self._ratio: float | None = None

    @property
    def ratio(self) -> float:
        """Provider tokens per local token; 1.0 until a report has been observed."""
        return self._ratio or 1.0

    def estimate(self, local_message_tokens: int) -> int:
        """Provider tokens for a request whose messages count ``local_message_tokens`` locally."""
        if self._ratio is None:
            return local_message_tokens
        return math.ceil((local_message_tokens + self._tool_tokens) * self._ratio)

    def observe(self, local_message_tokens: int, reported_input_tokens: object) -> None:
        """Learn from a sent request: its local message count and the input tokens the provider reported.

        A missing or non-positive report (a provider that returns no usage) teaches
        nothing and leaves the previous estimate in place.
        """
        if not isinstance(reported_input_tokens, int) or reported_input_tokens <= 0:
            return
        sent = local_message_tokens + self._tool_tokens
        ratio = min(_MAX_TOKENIZER_RATIO, max(1.0, reported_input_tokens / sent)) if sent > 0 else 1.0
        self._ratio = ratio if self._ratio is None else max(self._ratio, ratio)

    def to_local(self, provider_tokens: int) -> int:
        """Convert a provider-token allowance into the local-encoding tokens budgets are counted in."""
        return int(provider_tokens / self.ratio)
