"""Адаптер ART/αNLI: данные и промпты без передачи эталонных меток в LLM."""

from dataclasses import dataclass
import hashlib
import json
import random
import re

DATASET = "allenai/art"
CONFIG = "anli"
SPLIT = "validation"
PROMPT_VERSION = "anli-direct-v1"


@dataclass(frozen=True)
class ANLICase:
    index: int
    observation_1: str
    observation_2: str
    hypothesis_1: str
    hypothesis_2: str
    gold: int

    @classmethod
    def from_row(cls, index: int, row: dict):
        label = int(row["label"])
        if label not in (1, 2):
            raise ValueError(f"Пример {index}: ожидается label 1 или 2; получено {label}")
        fields = ("observation_1", "observation_2", "hypothesis_1", "hypothesis_2")
        values = [row[key] for key in fields]
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"Пример {index}: есть пустое текстовое поле")
        return cls(index, *values, label)


def select_indices(total: int, *, count: int, seed: int, offset: int = 0) -> list[int]:
    """Префикс для пилота и следующий сегмент для основного запуска не пересекаются."""
    if total < 1 or offset < 0 or count < 1 or offset + count > total:
        raise ValueError(f"Некорректная выборка: total={total}, count={count}, offset={offset}")
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return sorted(indices[offset:offset + count])


def load_validation():
    # Не требуется для unit-тестов и при импорте пакета агента.
    from datasets import load_dataset
    return load_dataset(DATASET, CONFIG, split=SPLIT)


def make_direct_prompt(case: ANLICase) -> str:
    """Gold намеренно не используется и в аргументы клиента не передаётся."""
    return (
        "Solve this abductive natural language inference task.\n"
        "Two events are observed in chronological order. Choose which proposed "
        "intermediate explanation most plausibly connects them.\n\n"
        f"Observation 1: {case.observation_1}\n"
        f"Observation 2: {case.observation_2}\n\n"
        f"H1: {case.hypothesis_1}\n"
        f"H2: {case.hypothesis_2}\n\n"
        "Compare both options using commonsense causality and event order. "
        "Do not invent additional observed evidence.\n"
        "End your response with exactly one line: ANSWER: H1 or ANSWER: H2"
    )


def parse_answer(text: str) -> int | None:
    """Чужие обозначения, несколько финальных меток и прочее — invalid output."""
    answers = []
    for line in text.splitlines():
        clean = line.strip().replace("**", "").strip("` ")
        match = re.fullmatch(r"ANSWER\s*:\s*H([12])\.?", clean, re.I)
        if match:
            answers.append(int(match.group(1)))
    return answers[0] if len(answers) == 1 else None


def cohort_hash(cases: list[ANLICase]) -> str:
    """Улавливает изменение данных между прерванным и возобновлённым запуском."""
    serial = json.dumps([vars(c) for c in cases], ensure_ascii=False,
                        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()
