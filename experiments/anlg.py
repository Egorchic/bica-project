"""Adapter for GEM/ART abductive natural language generation (alphaNLG).

The human bridge/reference is kept outside every prompt and is used only after
generation for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re

DATASET = "GEM/ART"
SPLIT = "validation"
PROMPT_VERSION = "anlg-bridge-v1"


@dataclass(frozen=True)
class ANLGCase:
    index: int
    gem_id: str
    observation_1: str
    observation_2: str
    references: tuple[str, ...]

    @classmethod
    def from_row(cls, index: int, row: dict) -> "ANLGCase":
        observation_1 = row.get("observation_1")
        observation_2 = row.get("observation_2")
        if not isinstance(observation_1, str) or not observation_1.strip():
            raise ValueError(f"Example {index}: empty observation_1")
        if not isinstance(observation_2, str) or not observation_2.strip():
            raise ValueError(f"Example {index}: empty observation_2")

        refs = row.get("references")
        if refs is None:
            target = row.get("target", row.get("label"))
            refs = [target] if isinstance(target, str) else []
        elif isinstance(refs, str):
            refs = [refs]
        else:
            refs = list(refs)

        refs = tuple(r.strip() for r in refs if isinstance(r, str) and r.strip())
        if not refs:
            target = row.get("target", row.get("label"))
            if isinstance(target, str) and target.strip():
                refs = (target.strip(),)
        if not refs:
            raise ValueError(f"Example {index}: no public reference text")

        gem_id = row.get("gem_id")
        if not isinstance(gem_id, str) or not gem_id.strip():
            gem_id = f"GEM-ART-{SPLIT}-{index}"
        return cls(index, gem_id.strip(), observation_1.strip(),
                   observation_2.strip(), refs)


def select_indices(total: int, *, count: int, seed: int, offset: int = 0) -> list[int]:
    if total < 1 or offset < 0 or count < 1 or offset + count > total:
        raise ValueError(
            f"Invalid sample: total={total}, count={count}, offset={offset}"
        )
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return sorted(indices[offset:offset + count])


def load_validation():
    from datasets import load_dataset
    return load_dataset(DATASET, revision="refs/convert/parquet", split=SPLIT)


def make_direct_prompt(case: ANLGCase) -> str:
    return (
        "Solve this abductive natural language generation task.\n"
        "A beginning event and a later ending event are observed. Generate ONE "
        "plausible intermediate event that could have happened after Observation 1 "
        "and before Observation 2, making the three-event story causally and "
        "chronologically coherent.\n\n"
        f"Observation 1: {case.observation_1}\n"
        f"Observation 2: {case.observation_2}\n\n"
        "Do not copy either observation. Do not mention answer keys or references. "
        "Prefer one concise natural-language sentence and avoid unnecessary details.\n"
        "End your response with exactly one line:\n"
        "BRIDGE: <one plausible intermediate event>"
    )


def parse_bridge(text: str) -> str | None:
    bridges = []
    for line in text.splitlines():
        clean = line.strip().replace("**", "").strip("` ")
        match = re.fullmatch(r"BRIDGE\s*:\s*(.+)", clean, re.I)
        if match and match.group(1).strip():
            bridges.append(match.group(1).strip())
    return bridges[0] if len(bridges) == 1 else None


def cohort_hash(cases: list[ANLGCase]) -> str:
    serial = json.dumps(
        [
            {
                "index": c.index,
                "gem_id": c.gem_id,
                "observation_1": c.observation_1,
                "observation_2": c.observation_2,
                "references": list(c.references),
            }
            for c in cases
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()
