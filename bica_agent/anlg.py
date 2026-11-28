"""Task-specific free-generation adapter for alphaNLG.

It reuses the same AbductiveAgent mechanisms as the main architecture but makes
the hypothesis-generation prompt explicitly produce bridge events between two
ordered observations.
"""

from __future__ import annotations

import re

from .core import AbductiveAgent


class AlphaNLGAgent(AbductiveAgent):
    """AbductiveAgent with a bridge-generation prompt tailored to alphaNLG."""

    def _generate(self, scenario: dict, round_no: int, trace: list):
        n = self.config.n_generate
        observations = scenario["observations"]
        if len(observations) != 2:
            raise ValueError("alphaNLG requires exactly two ordered observations")

        template = "\n".join(f"H{i}: one-sentence bridge event" for i in range(1, n + 1))
        prompt = f"""Generate exactly {n} distinct candidate bridge events.

Observation 1 (earlier):
{observations[0]}

Observation 2 (later):
{observations[1]}

Each candidate must describe ONE plausible event that happens after Observation 1
and before Observation 2. The candidate should make the complete story causally
and chronologically coherent. Use a genuinely different causal mechanism for
each candidate where possible. Do not rank the candidates. Do not invent facts
that are presented as already observed, and do not copy either observation.

Output exactly these {n} lines and nothing else:
{template}

Replace "one-sentence bridge event" with the candidate text."""
        text = self._ask(
            prompt,
            seed=100 + round_no,
            stage="generate bridge candidates",
            trace=trace,
        )

        found = {}
        for line in text.splitlines():
            cleaned = line.strip().replace("**", "").strip("` ")
            match = re.fullmatch(r"(H\d+)\s*:\s*(.+)", cleaned, re.I)
            if match:
                key = match.group(1).upper()
                if key in found:
                    raise ValueError("Duplicate hypothesis ID in alphaNLG generation")
                value = match.group(2).strip()
                if value:
                    found[key] = value

        expected = [f"H{i}" for i in range(1, n + 1)]
        if list(found) != expected:
            raise ValueError(f"Invalid alphaNLG candidate format: {text[:1000]}")
        scenario["hypotheses"] = found
        self.store.save(scenario)
