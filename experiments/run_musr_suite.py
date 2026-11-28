"""MuSR: compact transfer evaluation for BICA.

M0 = one direct model answer.
M1 = structured evidence ledger + three selection votes + self-refine.

Gold answers are attached only after every model call for a case is complete.
Each completed case is saved atomically, so the run is resumable.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from bica_agent import OllamaLLM
from bica_agent.client import IncompleteLLMResponse
from bica_agent.musr import MuSRReasoner
from .musr import (
    DATASET,
    DATASET_CONFIG,
    DATASET_REVISION,
    DOMAINS,
    MuSRCase,
    cohort_hash,
    load_cases_by_domain,
    select_cases,
)

PROTOCOL = "BICA-MUSR-M0-M1-v1"
PROFILES = {
    "M0": {
        "method": "direct",
        "description": "single direct answer",
    },
    "M1": {
        "method": "structured",
        "description": "evidence ledger + 3 votes + self-refine",
    },
}


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    paths = {
        "suite": root / "experiments" / "run_musr_suite.py",
        "dataset_adapter": root / "experiments" / "musr.py",
        "musr_reasoner": root / "bica_agent" / "musr.py",
        "ollama_client": root / "bica_agent" / "client.py",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def manifest_for(
    cases: list[MuSRCase],
    *,
    model: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    sample_seed: int,
    generation_seed: int,
    count: int,
    offset: int,
) -> dict:
    by_domain = Counter(c.domain for c in cases)
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "splits": list(DOMAINS),
        "model": model,
        "temperature": temperature,
        "think": True,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "sample_seed": sample_seed,
        "generation_seed": generation_seed,
        "count": count,
        "offset": offset,
        "uids": [c.uid for c in cases],
        "count_by_domain": dict(by_domain),
        "cohort_sha256": cohort_hash(cases),
        "profiles": PROFILES,
        "source_sha256": code_fingerprints(),
    }


def ensure_manifest(path: Path, proposed: dict) -> None:
    if path.exists():
        if read_json(path) != proposed:
            raise ValueError(
                "Конфигурация, выборка или протокол изменились. "
                "Используйте тот же код/параметры или новый --run-id."
            )
    else:
        atomic_json(path, proposed)


class MeteredLLM:
    def __init__(self, base):
        self.base = base
        self.model = getattr(base, "model", "unknown")
        self.records: list[dict] = []

    def ask(self, prompt: str, *, seed: int, stage: str):
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        def record(reply, incomplete: bool):
            self.records.append(
                {
                    "stage": stage,
                    "seed": seed,
                    "prompt_sha256": digest,
                    "response": reply.text,
                    "seconds": reply.seconds,
                    "output_tokens": reply.tokens,
                    "done_reason": reply.done_reason,
                    "incomplete": incomplete,
                }
            )

        print(f"    [LLM] {stage}: start", flush=True)
        try:
            reply = self.base.ask(prompt, seed=seed, stage=stage)
        except IncompleteLLMResponse as exc:
            record(exc.reply, True)
            print(
                f"    [LLM] {stage}: {exc.reply.done_reason} | "
                f"{exc.reply.seconds:.1f}s | {exc.reply.tokens} tokens",
                flush=True,
            )
            raise
        record(reply, False)
        print(
            f"    [LLM] {stage}: {reply.done_reason} | "
            f"{reply.seconds:.1f}s | {reply.tokens} tokens",
            flush=True,
        )
        return reply


def stable_seed_offset(uid: str) -> int:
    return int(hashlib.sha256(uid.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000


def run_one(
    case: MuSRCase,
    *,
    config: str,
    base_llm,
    generation_seed: int,
) -> dict:
    metered = MeteredLLM(base_llm)
    reasoner = MuSRReasoner(metered)
    seed_base = generation_seed + stable_seed_offset(case.uid)
    details = {}
    prediction = None
    invalid_reason = None

    try:
        kwargs = dict(
            domain=case.domain,
            context=case.context,
            question=case.question,
            choices=case.choices,
            seed_base=seed_base,
        )
        if config == "M0":
            details = reasoner.direct(**kwargs)
        elif config == "M1":
            details = reasoner.structured(**kwargs)
        else:
            raise ValueError(f"Unknown config: {config}")
        prediction = details["prediction"]
    except IncompleteLLMResponse as exc:
        invalid_reason = (
            "truncated" if exc.reply.done_reason == "length"
            else "incomplete_response"
        )
    except ValueError as exc:
        invalid_reason = "invalid_format"
        details = {"error": str(exc)[:1000]}

    valid = invalid_reason is None and isinstance(prediction, int)
    if valid and not 0 <= prediction < len(case.choices):
        valid = False
        invalid_reason = "invalid_prediction"
        prediction = None

    # Gold is read only after every LLM call above has returned.
    gold = case.gold
    preliminary = (
        details.get("preliminary_prediction")
        if isinstance(details, dict)
        else None
    )

    return {
        "uid": case.uid,
        "domain": case.domain,
        "source_index": case.source_index,
        "question_index": case.question_index,
        "config": config,
        "prediction": prediction if valid else None,
        "prediction_number": prediction + 1 if valid else None,
        "preliminary_prediction": preliminary,
        "gold": gold,
        "gold_number": gold + 1,
        "correct": bool(valid and prediction == gold),
        "valid": valid,
        "invalid_reason": invalid_reason,
        "seconds": round(sum(x["seconds"] for x in metered.records), 2),
        "output_tokens": sum(x["output_tokens"] for x in metered.records),
        "llm_calls": len(metered.records),
        "calls": metered.records,
        "details": details,
    }


def load_results(out_dir: Path, config: str, uids: list[str]) -> list[dict]:
    results = []
    for uid in uids:
        filename = uid.replace(":", "__") + ".json"
        path = out_dir / config / "predictions" / filename
        if path.exists():
            results.append(read_json(path))
    return results


def wilson_interval(success: int, count: int, z: float = 1.95996398454):
    if count == 0:
        return None
    p = success / count
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    spread = z * math.sqrt(
        p * (1 - p) / count + z * z / (4 * count * count)
    ) / denom
    return [
        round(max(0.0, center - spread), 6),
        round(min(1.0, center + spread), 6),
    ]


def _normalized_accuracy(acc: float | None, chance: float) -> float | None:
    if acc is None:
        return None
    if acc <= chance:
        return 0.0
    return round((acc - chance) / (1.0 - chance), 6)


def summary_for(items: list[dict], selected_cases: list[MuSRCase]) -> dict:
    n = len(items)
    correct = sum(bool(x["correct"]) for x in items)
    valid = sum(bool(x["valid"]) for x in items)

    by_domain = {}
    domain_accuracies = []
    normalized = []
    for domain in DOMAINS:
        subset = [x for x in items if x["domain"] == domain]
        expected_selected = sum(c.domain == domain for c in selected_cases)
        n_d = len(subset)
        correct_d = sum(bool(x["correct"]) for x in subset)
        acc = round(correct_d / n_d, 6) if n_d else None
        selected_domain_cases = [c for c in selected_cases if c.domain == domain]
        chance = (
            sum(1.0 / len(c.choices) for c in selected_domain_cases)
            / len(selected_domain_cases)
            if selected_domain_cases else None
        )
        norm = _normalized_accuracy(acc, chance) if chance is not None else None
        choice_counts = Counter(len(c.choices) for c in selected_domain_cases)
        by_domain[domain] = {
            "selected": expected_selected,
            "processed": n_d,
            "complete": n_d == expected_selected,
            "correct": correct_d,
            "accuracy": acc,
            "sample_mean_random_chance": (
                round(chance, 6) if chance is not None else None
            ),
            "choice_count_distribution": {
                str(k): v for k, v in sorted(choice_counts.items())
            },
            "sample_chance_normalized_accuracy": norm,
        }
        if acc is not None:
            domain_accuracies.append(acc)
        if norm is not None:
            normalized.append(norm)

    reflection = {
        "revise_verdicts": 0,
        "changed_after_revision": 0,
        "wrong_to_right": 0,
        "right_to_wrong": 0,
    }
    for x in items:
        if x["config"] != "M1" or not x["valid"]:
            continue
        d = x.get("details", {})
        if d.get("critique", {}).get("verdict") != "REVISE":
            continue
        reflection["revise_verdicts"] += 1
        before = d.get("preliminary_prediction")
        after = x["prediction"]
        if before is None:
            continue
        reflection["changed_after_revision"] += int(before != after)
        reflection["wrong_to_right"] += int(before != x["gold"] and after == x["gold"])
        reflection["right_to_wrong"] += int(before == x["gold"] and after != x["gold"])

    return {
        "selected": len(selected_cases),
        "processed": n,
        "complete": n == len(selected_cases),
        "correct": correct,
        "valid": valid,
        "invalid": n - valid,
        "truncated": sum(x.get("invalid_reason") == "truncated" for x in items),
        "accuracy": round(correct / n, 6) if n else None,
        "wilson_95": wilson_interval(correct, n),
        "macro_accuracy": (
            round(sum(domain_accuracies) / len(domain_accuracies), 6)
            if domain_accuracies else None
        ),
        "chance_normalized_macro": (
            round(sum(normalized) / len(normalized), 6)
            if normalized else None
        ),
        "by_domain": by_domain,
        "total_seconds": round(sum(x["seconds"] for x in items), 2),
        "mean_seconds": (
            round(sum(x["seconds"] for x in items) / n, 2) if n else None
        ),
        "output_tokens": sum(x["output_tokens"] for x in items),
        "llm_calls": sum(x["llm_calls"] for x in items),
        "reflection": reflection,
    }


def exact_mcnemar(a: list[dict], b: list[dict]) -> dict:
    map_a = {x["uid"]: x for x in a}
    map_b = {x["uid"]: x for x in b}
    if set(map_a) != set(map_b):
        raise ValueError("McNemar requires identical MuSR case IDs")
    a_only = sum(
        map_a[k]["correct"] and not map_b[k]["correct"] for k in map_a
    )
    b_only = sum(
        map_b[k]["correct"] and not map_a[k]["correct"] for k in map_a
    )
    discordant = a_only + b_only
    if discordant == 0:
        p = 1.0
    else:
        low = min(a_only, b_only)
        tail = sum(
            math.comb(discordant, i) * (0.5 ** discordant)
            for i in range(low + 1)
        )
        p = min(1.0, 2 * tail)
    return {
        "only_M0_correct": a_only,
        "only_M1_correct": b_only,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": round(p, 8),
    }


def report(out_dir: Path, cases: list[MuSRCase]) -> dict:
    uids = [c.uid for c in cases]
    summaries = {}
    all_results = {}
    for config in PROFILES:
        items = load_results(out_dir, config, uids)
        if not items:
            continue
        all_results[config] = items
        summaries[config] = summary_for(items, cases)
        atomic_json(out_dir / config / "summary.json", summaries[config])

    comparison = None
    if (
        "M0" in summaries
        and "M1" in summaries
        and summaries["M0"]["complete"]
        and summaries["M1"]["complete"]
    ):
        comparison = exact_mcnemar(all_results["M0"], all_results["M1"])

    result = {
        "protocol": PROTOCOL,
        "by_config": summaries,
        "M0_vs_M1": comparison,
    }
    atomic_json(out_dir / "comparison.json", result)

    columns = [
        "config",
        "processed",
        "correct",
        "accuracy",
        "macro_accuracy",
        "chance_normalized_macro",
        "invalid",
        "truncated",
        "mean_seconds",
        "llm_calls",
    ]
    with (out_dir / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for config, s in summaries.items():
            writer.writerow(
                {"config": config, **{k: s[k] for k in columns if k != "config"}}
            )
    return result


def run_suite(
    cases: list[MuSRCase],
    *,
    base_llm,
    output_dir: Path,
    configs: list[str],
    generation_seed: int,
) -> dict:
    out_dir = Path(output_dir)
    for config in configs:
        print(f"\n=== {config}: {PROFILES[config]['description']} ===", flush=True)
        for position, case in enumerate(cases, 1):
            filename = case.uid.replace(":", "__") + ".json"
            path = out_dir / config / "predictions" / filename
            if path.exists():
                print(
                    f"[{config} {position}/{len(cases)}] "
                    f"{case.uid}: already saved",
                    flush=True,
                )
                continue

            print(
                f"[{config} {position}/{len(cases)}] "
                f"{case.uid}: generating...",
                flush=True,
            )
            item = run_one(
                case,
                config=config,
                base_llm=base_llm,
                generation_seed=generation_seed,
            )
            atomic_json(path, item)
            print(
                f"  answer={item['prediction_number']} "
                f"| gold={item['gold_number']} "
                f"| correct={item['correct']} "
                f"| valid={item['valid']} "
                f"| {item['seconds']:.1f}s "
                f"| calls={item['llm_calls']}",
                flush=True,
            )
            if position % 5 == 0 or position == len(cases):
                report(out_dir, cases)
    return report(out_dir, cases)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MuSR M0/M1 transfer evaluation"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--run-id", default="musr_main_50")
    parser.add_argument("--configs", default="M0,M1")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=2026)
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--temperature", type=float, default=0.6)
    # MuSR stories are long; 8192 avoids truncating story + structured feedback.
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=2400)
    parser.add_argument("--results-dir", default="results/musr")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id can only include letters, digits, _ and -")
    configs = [
        x.strip().upper() for x in args.configs.split(",") if x.strip()
    ]
    if (
        not configs
        or len(set(configs)) != len(configs)
        or any(c not in PROFILES for c in configs)
    ):
        parser.error("--configs must contain unique values among M0,M1")

    print("Loading TAUR-Lab/MuSR...", flush=True)
    cases_by_domain = load_cases_by_domain()
    cases = select_cases(
        cases_by_domain,
        count=args.count,
        offset=args.offset,
        seed=args.sample_seed,
    )
    print(
        "Selected:",
        dict(Counter(c.domain for c in cases)),
        flush=True,
    )

    out_dir = Path(args.results_dir) / args.run_id
    manifest = manifest_for(
        cases,
        model=args.model,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        sample_seed=args.sample_seed,
        generation_seed=args.generation_seed,
        count=args.count,
        offset=args.offset,
    )
    try:
        ensure_manifest(out_dir / "manifest.json", manifest)
        model = OllamaLLM(
            model=args.model,
            host=args.host,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )
        result = run_suite(
            cases,
            base_llm=model,
            output_dir=out_dir,
            configs=configs,
            generation_seed=args.generation_seed,
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted; completed cases are saved. "
            "Rerun the exact same command.",
            file=sys.stderr,
            flush=True,
        )
        return 130
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 2

    print("\nFinished. Current comparison:", flush=True)
    for config, s in result["by_config"].items():
        print(
            f"{config}: {s['correct']}/{s['processed']} "
            f"accuracy={s['accuracy']} "
            f"macro={s['macro_accuracy']} "
            f"norm_macro={s['chance_normalized_macro']} "
            f"invalid={s['invalid']}",
            flush=True,
        )
    print("Saved to:", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
