"""Adapter for the MuSR benchmark.

The Hugging Face mirror TAUR-Lab/MuSR exposes three task CSV files. We load
those files directly instead of using the dataset card's `all` split because
newer versions of `datasets` reserve `all` as a special split expression.

Gold answers are stored in MuSRCase but are never used by prompt construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
import random

DATASET = "TAUR-Lab/MuSR"
DATASET_CONFIG = "direct_csv"
DATASET_REVISION = "3ef7d9886a882d5b68c5b54225d1a423be7ddf36"
DOMAINS = ("murder_mysteries", "object_placements", "team_allocation")
DOMAIN_FILES = {
    "murder_mysteries": "murder_mystery.csv",
    "object_placements": "object_placements.csv",
    "team_allocation": "team_allocation.csv",
}


@dataclass(frozen=True)
class MuSRCase:
    uid: str
    domain: str
    source_index: int
    question_index: int
    context: str
    question: str
    choices: tuple[str, ...]
    gold: int  # zero-based index


def _parse_choices(value) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        choices = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty choices field")
        # HF CSV stores Python-style lists such as ['Alice', 'Bob'].
        try:
            choices = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                choices = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Cannot parse choices: {text[:200]!r}") from exc
    else:
        raise ValueError(f"Unexpected choices type: {type(value).__name__}")

    if not isinstance(choices, (list, tuple)):
        raise ValueError("Parsed choices must be a list/tuple")
    return tuple(str(x).strip() for x in choices)


def _flat_case(domain: str, source_index: int, row: dict) -> MuSRCase:
    """Parse the current TAUR-Lab/MuSR CSV schema."""
    context = row.get("narrative")
    question = row.get("question")
    choices = _parse_choices(row.get("choices"))
    gold = row.get("answer_index")

    if not isinstance(context, str) or not context.strip():
        raise ValueError(f"{domain}[{source_index}]: missing narrative")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{domain}[{source_index}]: missing question")

    if len(choices) < 2 or any(not x for x in choices):
        raise ValueError(
            f"{domain}[{source_index}]: expected at least 2 non-empty choices, "
            f"got {len(choices)}"
        )

    try:
        gold = int(gold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{domain}[{source_index}]: invalid answer_index") from exc

    if not 0 <= gold < len(choices):
        raise ValueError(
            f"{domain}[{source_index}]: answer_index {gold} outside "
            f"0..{len(choices)-1}"
        )

    # Optional consistency check against answer_choice from the HF mirror.
    answer_choice = row.get("answer_choice")
    if (
        isinstance(answer_choice, str)
        and answer_choice.strip()
        and answer_choice.strip() != choices[gold]
    ):
        raise ValueError(
            f"{domain}[{source_index}]: answer_choice does not match answer_index"
        )

    return MuSRCase(
        uid=f"{domain}:{source_index}:0",
        domain=domain,
        source_index=source_index,
        question_index=0,
        context=context.strip(),
        question=question.strip(),
        choices=choices,
        gold=gold,
    )


def _nested_questions_from_row(row: dict) -> list[dict]:
    """Compatibility parser for the original MuSR JSON schema."""
    value = row.get("questions")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty questions field")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = ast.literal_eval(text)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"Unexpected questions field: {type(value).__name__}")
    if not all(isinstance(x, dict) for x in value):
        raise ValueError("Each question must be an object")
    return value


def cases_from_rows(domain: str, rows) -> list[MuSRCase]:
    """Parse either the current HF CSV schema or the original nested JSON."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown MuSR domain: {domain}")

    result: list[MuSRCase] = []
    for source_index, row in enumerate(rows):
        # Current TAUR-Lab/MuSR Hugging Face CSV.
        if "narrative" in row and "answer_index" in row:
            result.append(_flat_case(domain, source_index, row))
            continue

        # Original GitHub JSON compatibility.
        context = row.get("context")
        if not isinstance(context, str) or not context.strip():
            raise ValueError(f"{domain}[{source_index}]: missing context")

        for question_index, item in enumerate(_nested_questions_from_row(row)):
            question = item.get("question")
            choices = _parse_choices(item.get("choices"))
            gold = item.get("answer")

            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"{domain}[{source_index}] q{question_index}: missing question"
                )
            if len(choices) < 2 or any(not x for x in choices):
                raise ValueError(
                    f"{domain}[{source_index}] q{question_index}: "
                    f"expected at least 2 non-empty choices, got {len(choices)}"
                )
            try:
                gold = int(gold)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{domain}[{source_index}] q{question_index}: invalid answer"
                ) from exc
            if not 0 <= gold < len(choices):
                raise ValueError(
                    f"{domain}[{source_index}] q{question_index}: "
                    f"answer {gold} outside 0..{len(choices)-1}"
                )

            result.append(
                MuSRCase(
                    uid=f"{domain}:{source_index}:{question_index}",
                    domain=domain,
                    source_index=source_index,
                    question_index=question_index,
                    context=context.strip(),
                    question=question.strip(),
                    choices=choices,
                    gold=gold,
                )
            )
    return result


def _pinned_csv_url(filename: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET}/resolve/"
        f"{DATASET_REVISION}/{filename}"
    )


def load_cases_by_domain() -> dict[str, list[MuSRCase]]:
    """Load each MuSR domain directly from its pinned CSV.

    This intentionally avoids `load_dataset("TAUR-Lab/MuSR", ...)` because the
    dataset metadata also declares a split named `all`, which collides with the
    special `all` split keyword in newer versions of Hugging Face `datasets`.
    """
    from datasets import load_dataset

    result = {}
    for domain in DOMAINS:
        filename = DOMAIN_FILES[domain]
        rows = load_dataset(
            "csv",
            data_files=_pinned_csv_url(filename),
            split="train",
        )
        result[domain] = cases_from_rows(domain, rows)
    return result


def balanced_order(cases_by_domain: dict[str, list[MuSRCase]], *, seed: int) -> list[MuSRCase]:
    """Deterministic round-robin order after an independent shuffle per domain.

    The first 50 cases therefore contain 17 murder, 17 object and 16 team cases.
    """
    pools: dict[str, list[MuSRCase]] = {}
    for domain_no, domain in enumerate(DOMAINS):
        pool = list(cases_by_domain[domain])
        random.Random(seed + 1009 * domain_no).shuffle(pool)
        pools[domain] = pool

    result: list[MuSRCase] = []
    pos = 0
    while True:
        added = False
        for domain in DOMAINS:
            if pos < len(pools[domain]):
                result.append(pools[domain][pos])
                added = True
        if not added:
            break
        pos += 1
    return result


def select_cases(
    cases_by_domain: dict[str, list[MuSRCase]],
    *,
    count: int = 50,
    offset: int = 0,
    seed: int = 42,
) -> list[MuSRCase]:
    ordered = balanced_order(cases_by_domain, seed=seed)
    if count < 1 or offset < 0 or offset + count > len(ordered):
        raise ValueError(
            f"Invalid MuSR sample: total={len(ordered)}, count={count}, offset={offset}"
        )
    return ordered[offset:offset + count]


def cohort_hash(cases: list[MuSRCase]) -> str:
    payload = [
        {
            "uid": c.uid,
            "domain": c.domain,
            "source_index": c.source_index,
            "question_index": c.question_index,
            "context": c.context,
            "question": c.question,
            "choices": list(c.choices),
            "gold": c.gold,
        }
        for c in cases
    ]
    serial = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()
