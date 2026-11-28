"""Belief-R dataset adapter for sequential belief-revision experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re

DATASET = "CAiRE/belief_r"
SPLIT = "test"
PROMPT_VERSION = "belief-r-sequential-v1"
QUESTION_TEXT = "What necessarily had to follow assuming that the above premises were true?"


@dataclass(frozen=True)
class BeliefRCase:
    index: int
    dataset_id: str
    atomic_idx: int
    modus: str
    relation: str  # BU or BM
    premise_1: str
    premise_2: str
    premise_3: str
    option_a: str
    option_b: str
    option_c: str
    gold_t: str
    gold_t1: str
    types_of_relation: str

    @property
    def options(self) -> list[str]:
        return [self.option_a, self.option_b, self.option_c]

    @classmethod
    def from_row(cls, index: int, row: dict) -> "BeliefRCase":
        step = str(row.get("step", "")).strip().lower()
        if step != "time_t1":
            raise ValueError(f"Belief-R row {index}: expected time_t1, got {step!r}")

        modus = str(row.get("modus", "")).strip().lower()
        if modus not in {"ponens", "tollens"}:
            raise ValueError(f"Belief-R row {index}: unknown modus {modus!r}")

        gold_t1 = str(row.get("ground_truth", "")).strip().lower()
        if gold_t1 not in {"a", "b", "c"}:
            raise ValueError(f"Belief-R row {index}: invalid ground truth {gold_t1!r}")

        # In the basic time_t query, modus ponens entails option a and modus
        # tollens entails option b by Belief-R construction.
        gold_t = "a" if modus == "ponens" else "b"
        if gold_t1 == "c":
            relation = "BU"
        elif gold_t1 == gold_t:
            relation = "BM"
        else:
            raise ValueError(
                f"Belief-R row {index}: unexpected t+1 gold {gold_t1!r} "
                f"for modus {modus!r}"
            )

        premises = extract_premises(str(row.get("questions", "")))
        if len(premises) != 3:
            raise ValueError(
                f"Belief-R row {index}: expected 3 premises at time_t1, "
                f"found {len(premises)}: {premises!r}"
            )

        options = [str(row.get(k, "")).strip() for k in ("a", "b", "c")]
        if any(not option for option in options):
            raise ValueError(f"Belief-R row {index}: empty answer option")

        return cls(
            index=index,
            dataset_id=str(row.get("dataset_id", index)),
            atomic_idx=int(row.get("atomic_idx", -1)),
            modus=modus,
            relation=relation,
            premise_1=premises[0],
            premise_2=premises[1],
            premise_3=premises[2],
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            gold_t=gold_t,
            gold_t1=gold_t1,
            types_of_relation=str(row.get("types_of_relation", "")),
        )


def extract_premises(question: str) -> list[str]:
    """Extract premises preceding Belief-R's fixed question sentence."""
    parts = re.split(
        r"What necessarily had to follow assuming that the above premises were true\?",
        question,
        maxsplit=1,
        flags=re.I,
    )
    if len(parts) != 2:
        raise ValueError(f"Belief-R question marker not found: {question[:300]!r}")
    prefix = parts[0].strip()
    return [line.strip() for line in prefix.splitlines() if line.strip()]


def load_time_t1_cases() -> list[BeliefRCase]:
    # Deferred import keeps unit tests independent from Hugging Face/network.
    from datasets import load_dataset

    rows = load_dataset(DATASET, split=SPLIT)
    cases = []
    for index, row in enumerate(rows):
        if str(row.get("step", "")).strip().lower() == "time_t1":
            cases.append(BeliefRCase.from_row(index, row))
    if not cases:
        raise ValueError("No time_t1 rows found in Belief-R")
    return cases


def select_balanced_cases(
    cases: list[BeliefRCase], *, count: int = 50, seed: int = 42
) -> list[BeliefRCase]:
    """Select a deterministic BU/BM and ponens/tollens balanced cohort.

    For count=50 this gives exactly:
      BU-ponens=13, BU-tollens=12, BM-ponens=12, BM-tollens=13.
    Therefore BU=BM=25 and ponens=tollens=25.
    """
    if count < 4 or count % 2:
        raise ValueError("Belief-R count must be an even integer >= 4")

    per_relation = count // 2
    half = per_relation // 2
    extra = per_relation % 2
    targets = {
        ("BU", "ponens"): half + extra,
        ("BU", "tollens"): half,
        ("BM", "ponens"): half,
        ("BM", "tollens"): half + extra,
    }

    buckets: dict[tuple[str, str], list[BeliefRCase]] = {
        key: [] for key in targets
    }
    for case in cases:
        key = (case.relation, case.modus)
        if key in buckets:
            buckets[key].append(case)

    rng = random.Random(seed)
    selected: list[BeliefRCase] = []
    for key in (("BU", "ponens"), ("BU", "tollens"),
                ("BM", "ponens"), ("BM", "tollens")):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        need = targets[key]
        if len(bucket) < need:
            raise ValueError(f"Not enough Belief-R cases in stratum {key}: {len(bucket)} < {need}")
        selected.extend(bucket[:need])

    return sorted(selected, key=lambda case: case.index)


def make_prompt(case: BeliefRCase, *, step: str) -> str:
    if step == "time_t":
        premises = [case.premise_1, case.premise_2]
    elif step == "time_t1":
        premises = [case.premise_1, case.premise_2, case.premise_3]
    else:
        raise ValueError(f"Unknown Belief-R step: {step}")

    premise_text = "\n".join(
        f"Premise {i}: {premise}" for i, premise in enumerate(premises, 1)
    )
    return f"""{premise_text}

{QUESTION_TEXT}
A: {case.option_a}
B: {case.option_b}
C: {case.option_c}

Reason using the premises and commonsense causal dependencies. In this benchmark,
a newly introduced conditional premise may reveal an additional requirement for
a previous conclusion, or it may provide an independent alternative route.
Do not use any answer key or dataset metadata.
End with exactly one line: ANSWER: A, ANSWER: B, or ANSWER: C"""


def parse_answer(text: str) -> str | None:
    answers = []
    for line in text.splitlines():
        clean = line.strip().replace("**", "").strip("` ")
        match = re.fullmatch(r"ANSWER\s*:\s*([ABC])\.?", clean, re.I)
        if match:
            answers.append(match.group(1).lower())
    return answers[0] if len(answers) == 1 else None


def cohort_hash(cases: list[BeliefRCase]) -> str:
    serial = json.dumps(
        [vars(case) for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()
