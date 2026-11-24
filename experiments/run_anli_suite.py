"""ART/alphaNLI: C0-C5 evaluation without modifying the BICA agent.

Gold labels are used exclusively in result annotation and reporting after calls.
The sample cohort, model settings and code protocol are locked per run ID.
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

from bica_agent import AbductiveAgent, AgentConfig, OllamaLLM, ScenarioStore
from bica_agent.client import IncompleteLLMResponse
from .anli import (ANLICase, CONFIG, DATASET, PROMPT_VERSION, SPLIT,
                   cohort_hash, load_validation, make_direct_prompt,
                   parse_answer, select_indices)

PROTOCOL = "BICA-ART-ANLI-C0-C5-v1"
PROFILES = {
    "C0": {"method": "direct", "k": 1, "evaluate": False, "self_refine": False},
    "C1": {"method": "direct", "k": 3, "evaluate": False, "self_refine": False},
    "C2": {"method": "agent", "k": 1, "evaluate": True, "self_refine": False},
    "C3": {"method": "agent", "k": 1, "evaluate": True, "self_refine": True},
    "C4": {"method": "agent", "k": 3, "evaluate": True, "self_refine": False},
    "C5": {"method": "agent", "k": 3, "evaluate": True, "self_refine": True},
}


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_fingerprints() -> dict[str, str]:
    """Detect source changes between interrupted and resumed runs."""
    root = Path(__file__).resolve().parent.parent
    paths = {
        "suite": root / "experiments" / "run_anli_suite.py",
        "dataset_adapter": root / "experiments" / "anli.py",
        "agent_core": root / "bica_agent" / "core.py",
        "ollama_client": root / "bica_agent" / "client.py",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()}


def manifest_for(cases: list[ANLICase], *, model: str, temperature: float,
                 num_ctx: int, num_predict: int, sample_seed: int,
                 generation_seed: int, offset: int) -> dict:
    return {
        "protocol": PROTOCOL, "dataset": DATASET, "dataset_config": CONFIG,
        "split": SPLIT, "prompt_version": PROMPT_VERSION,
        "model": model, "temperature": temperature, "think": True,
        "num_ctx": num_ctx, "num_predict": num_predict,
        "sample_seed": sample_seed, "generation_seed": generation_seed,
        "count": len(cases), "offset": offset,
        "indices": [c.index for c in cases], "cohort_sha256": cohort_hash(cases),
        "profiles": PROFILES, "source_sha256": code_fingerprints(),
    }


def ensure_manifest(path: Path, proposed: dict) -> None:
    if path.exists():
        if read_json(path) != proposed:
            raise ValueError(
                "Конфигурация, выборка или протокол изменились. "
                "Создайте новый --run-id; не смешивайте разные эксперименты."
            )
    else:
        atomic_json(path, proposed)


class MeteredLLM:
    """Capture every completed call, including calls ending at a token limit."""

    def __init__(self, base, *, agent_seed_offset: int):
        self.base = base
        self.model = getattr(base, "model", "fake_llm")
        self.agent_seed_offset = agent_seed_offset
        self.records: list[dict] = []

    def ask(self, prompt: str, *, seed: int, stage: str):
        # Agent's internal seed bases (1100, 2100, ...) become case-specific.
        # Direct baseline sends its seed exactly as in run_anli_baseline.py.
        actual_seed = (seed if stage.startswith("anli_direct")
                       else seed + self.agent_seed_offset)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        def record(reply, *, incomplete: bool):
            self.records.append({
                "stage": stage, "seed": actual_seed,
                "prompt_sha256": digest, "response": reply.text,
                "seconds": reply.seconds, "output_tokens": reply.tokens,
                "done_reason": reply.done_reason, "incomplete": incomplete,
            })

        print(f"    [LLM] {stage}: start", flush=True)
        try:
            reply = self.base.ask(prompt, seed=actual_seed, stage=stage)
        except IncompleteLLMResponse as exc:
            record(exc.reply, incomplete=True)
            print(f"    [LLM] {stage}: {exc.reply.done_reason} | "
                  f"{exc.reply.seconds:.1f}s | {exc.reply.tokens} tokens", flush=True)
            raise
        record(reply, incomplete=False)
        print(f"    [LLM] {stage}: {reply.done_reason} | "
              f"{reply.seconds:.1f}s | {reply.tokens} tokens", flush=True)
        return reply


class InvalidOutput(ValueError):
    pass


def choice_as_int(hypothesis: str) -> int:
    if hypothesis not in ("H1", "H2"):
        raise InvalidOutput(f"Expected H1/H2, got {hypothesis!r}")
    return int(hypothesis[1])


def run_one(case: ANLICase, *, config: str, base_llm, state_dir: Path,
            generation_seed: int) -> dict:
    profile = PROFILES[config]
    metered = MeteredLLM(base_llm, agent_seed_offset=generation_seed + case.index)
    prediction = None
    before = None
    details: dict = {}
    invalid_reason = None
    try:
        if profile["method"] == "direct":
            prompt = make_direct_prompt(case)  # identical to the original C0 prompt
            votes: list[int] = []
            for vote_no in range(profile["k"]):
                reply = metered.ask(
                    prompt, seed=generation_seed + case.index + vote_no,
                    stage=f"anli_direct_vote_{vote_no + 1}",
                )
                answer = parse_answer(reply.text)
                if answer is None:
                    raise InvalidOutput("Missing or ambiguous ANSWER: H1/H2")
                votes.append(answer)
            counts = Counter(votes)
            prediction = counts.most_common(1)[0][0]
            before = prediction
            details = {"votes": votes, "vote_counts": dict(counts)}
        else:
            store = ScenarioStore(state_dir)
            state = store.create_or_load(
                f"case_{case.index:05d}",
                [case.observation_1, case.observation_2],
                [case.hypothesis_1, case.hypothesis_2],
            )
            agent = AbductiveAgent(
                metered, store,
                AgentConfig(
                    k=profile["k"], evaluate=profile["evaluate"],
                    self_refine=profile["self_refine"], history=False,
                ),
            )
            round_result = agent.run_round(state)
            before = choice_as_int(round_result["selection"]["chosen"])
            prediction = choice_as_int(round_result["final"])
            details = {
                "evaluations": round_result["evaluations"],
                "selection": round_result["selection"],
                "critique": round_result["critique"],
                "revision": round_result["revision"],
            }
            # If the app was interrupted between saving scenario memory and
            # writing prediction JSON, use persisted calls to recover costs.
            if not metered.records and round_result["calls"]:
                metered.records = [{
                    "stage": r["stage"], "seed": None,
                    "prompt_sha256": None, "response": r["response"],
                    "seconds": r["seconds"], "output_tokens": r["tokens"],
                    "done_reason": r["done_reason"],
                    "incomplete": False, "recovered_from_scenario": True,
                } for r in round_result["calls"]]
    except IncompleteLLMResponse as exc:
        invalid_reason = ("truncated" if exc.reply.done_reason == "length"
                          else "incomplete_response")
    except ValueError as exc:
        # Only the actual agent execution is wrapped. Misconfigured scenarios
        # or wrong input files are not silently treated as model failures.
        invalid_reason = "invalid_format"
        details["error"] = str(exc)[:700]

    valid = invalid_reason is None and prediction in (1, 2)
    return {
        "index": case.index, "config": config,
        "prediction": prediction if valid else None,
        "preliminary_prediction": before,
        # Gold is attached *only after* every call to the LLM has returned.
        "gold": case.gold, "correct": bool(valid and prediction == case.gold),
        "valid": valid, "invalid_reason": invalid_reason,
        "seconds": round(sum(c["seconds"] for c in metered.records), 2),
        "output_tokens": sum(c["output_tokens"] for c in metered.records),
        "llm_calls": len(metered.records),
        "calls": metered.records, "details": details,
    }


def import_c0(baseline_dir: Path, suite_dir: Path, cases: list[ANLICase],
              manifest: dict) -> int:
    """Reuse the original direct C0 pilot only if *all* settings match."""
    existing = read_json(baseline_dir / "manifest.json")
    keys = (
        "dataset", "dataset_config", "split", "prompt_version", "model",
        "temperature", "num_ctx", "num_predict", "think",
        "sample_seed", "generation_seed", "count", "offset", "indices",
        "cohort_sha256",
    )
    mismatch = [key for key in keys if existing.get(key) != manifest.get(key)]
    if mismatch:
        raise ValueError(f"Нельзя импортировать C0: не совпадают поля {mismatch}")

    for case in cases:
        source_path = baseline_dir / "predictions" / f"{case.index:05d}.json"
        if not source_path.exists():
            raise ValueError(f"Не найден результат C0: {source_path}")
        source = read_json(source_path)
        if source["index"] != case.index or source["gold"] != case.gold:
            raise ValueError(f"Некорректные поля index/gold в {source_path}")

    imported = 0
    for case in cases:
        target_path = suite_dir / "C0" / "predictions" / f"{case.index:05d}.json"
        if target_path.exists():
            continue
        source = read_json(baseline_dir / "predictions" / f"{case.index:05d}.json")
        if source.get("valid") and source.get("prediction") not in (1, 2):
            raise ValueError("Baseline C0 marked a non-answer as valid")
        item = {
            "index": case.index, "config": "C0",
            "prediction": source.get("prediction"),
            "preliminary_prediction": source.get("prediction"),
            "gold": case.gold, "correct": source["correct"],
            "valid": source["valid"],
            "invalid_reason": source.get("invalid_reason"),
            "seconds": source["seconds"],
            "output_tokens": source["output_tokens"],
            "llm_calls": source.get("calls", 1),
            "calls": [{
                "stage": "anli_direct_baseline", "seed": source.get("generation_seed"),
                "response": source.get("response", ""),
                "seconds": source["seconds"], "output_tokens": source["output_tokens"],
                "done_reason": source.get("done_reason"),
                "incomplete": source.get("invalid_reason") == "truncated",
            }],
            "details": {"imported_from": str(baseline_dir)},
        }
        atomic_json(target_path, item)
        imported += 1
    return imported


def load_results(out_dir: Path, config: str, indices: list[int]) -> list[dict]:
    return [read_json(p) for idx in indices
            if (p := out_dir / config / "predictions" / f"{idx:05d}.json").exists()]


def wilson_interval(success: int, count: int, z: float = 1.95996398454):
    if count == 0:
        return None
    p = success / count
    denom = 1 + z * z / count
    center = (p + z*z/(2*count)) / denom
    spread = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denom
    return [round(max(0.0, center - spread), 6),
            round(min(1.0, center + spread), 6)]


def summary_for(items: list[dict], n_selected: int) -> dict:
    n = len(items)
    correct = sum(bool(x["correct"]) for x in items)
    valid = sum(bool(x["valid"]) for x in items)
    invalid = n - valid
    corrected = degraded = kept_right = kept_wrong = revised = 0
    for r in items:
        if r["config"] not in ("C3", "C5") or not r["valid"]:
            continue
        before = r.get("preliminary_prediction")
        after = r["prediction"]
        if before is None:
            continue
        first = before == r["gold"]
        last = after == r["gold"]
        corrected += int(not first and last)
        degraded += int(first and not last)
        kept_right += int(first and last)
        kept_wrong += int(not first and not last)
        revised += int(r.get("details", {}).get("critique", {}).get("verdict") == "REVISE")
    seconds = sum(x["seconds"] for x in items)
    output_tokens = sum(x["output_tokens"] for x in items)
    calls = sum(x["llm_calls"] for x in items)
    return {
        "selected": n_selected, "processed": n, "complete": n == n_selected,
        "correct": correct, "valid": valid, "invalid": invalid,
        "truncated": sum(x.get("invalid_reason") == "truncated" for x in items),
        "accuracy_on_processed": round(correct/n, 6) if n else None,
        "wilson_95_on_processed": wilson_interval(correct, n),
        "total_seconds": round(seconds, 2),
        "mean_seconds": round(seconds/n, 2) if n else None,
        "output_tokens": output_tokens,
        "mean_output_tokens": round(output_tokens/n, 2) if n else None,
        "llm_calls": calls, "mean_llm_calls": round(calls/n, 2) if n else None,
        "reflection": {
            "wrong_to_right": corrected, "right_to_wrong": degraded,
            "right_to_right": kept_right, "wrong_to_wrong": kept_wrong,
            "revise_verdicts": revised,
            "note": "Only valid completed C3/C5 cases with a recorded preliminary choice.",
        },
    }


def exact_mcnemar(a: list[dict], b: list[dict]) -> dict:
    """Exact two-sided McNemar for identical fully evaluated cohorts."""
    map_a = {x["index"]: x for x in a}
    map_b = {x["index"]: x for x in b}
    if set(map_a) != set(map_b):
        raise ValueError("McNemar requires identical case IDs")
    a_only = sum(map_a[i]["correct"] and not map_b[i]["correct"] for i in map_a)
    b_only = sum(map_b[i]["correct"] and not map_a[i]["correct"] for i in map_a)
    discordant = a_only + b_only
    # Exact two-sided binomial tail, computed in log-space to handle even
    # the entire 1532-row validation split without integer/float overflow.
    if not discordant:
        p = 1.0
    else:
        lowest = min(a_only, b_only)
        logs = [math.lgamma(discordant + 1) - math.lgamma(i + 1)
                - math.lgamma(discordant - i + 1) - discordant * math.log(2)
                for i in range(lowest + 1)]
        largest = max(logs)
        p = min(1.0, 2 * math.exp(largest) * sum(
            math.exp(v - largest) for v in logs))
    return {"only_first_correct": a_only, "only_second_correct": b_only,
            "discordant": discordant, "mcnemar_exact_two_sided_p": round(p, 8)}


def report(out_dir: Path, indices: list[int]) -> dict:
    summaries = {}
    all_results = {}
    for config in PROFILES:
        items = load_results(out_dir, config, indices)
        if not items:
            continue
        all_results[config] = items
        summaries[config] = summary_for(items, len(indices))
        atomic_json(out_dir / config / "summary.json", summaries[config])

    comparisons = []
    pairs = [("C0", other) for other in ("C1", "C2", "C3", "C4", "C5")]
    pairs += [("C2", "C3"), ("C4", "C5"), ("C2", "C4"), ("C3", "C5")]
    for first, second in pairs:
        if first not in summaries or second not in summaries:
            continue
        if not summaries[first]["complete"] or not summaries[second]["complete"]:
            continue
        comparisons.append({"first": first, "second": second,
                            **exact_mcnemar(all_results[first], all_results[second])})

    results = {"protocol": PROTOCOL, "by_config": summaries,
               "paired_comparisons": comparisons}
    atomic_json(out_dir / "comparison.json", results)
    csv_path = out_dir / "comparison.csv"
    columns = ["config", "processed", "correct", "accuracy_on_processed",
               "wilson_95_on_processed", "invalid", "truncated",
               "mean_seconds", "mean_output_tokens", "mean_llm_calls"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for config, s in summaries.items():
            writer.writerow({"config": config, **{
                k: (str(s[k]) if k == "wilson_95_on_processed" else s[k])
                for k in columns if k != "config"
            }})
    return results


def run_suite(rows, *, base_llm, output_dir: Path, configs: list[str],
              baseline_dir: Path | None = None,
              count: int = 30, offset: int = 0, sample_seed: int = 42,
              generation_seed: int = 2026, model: str = "qwen3:14b",
              temperature: float = 0.6, num_ctx: int = 4096,
              num_predict: int = 2400) -> dict:
    if not configs and baseline_dir is None:
        raise ValueError("No configurations to run or baseline to import")
    if any(c not in PROFILES for c in configs):
        raise ValueError(f"Unknown configs: {configs}")
    indices = select_indices(len(rows), count=count, offset=offset, seed=sample_seed)
    cases = [ANLICase.from_row(i, rows[i]) for i in indices]
    out_dir = Path(output_dir)
    manifest = manifest_for(cases, model=model, temperature=temperature,
                            num_ctx=num_ctx, num_predict=num_predict,
                            sample_seed=sample_seed, generation_seed=generation_seed,
                            offset=offset)
    ensure_manifest(out_dir / "manifest.json", manifest)
    if baseline_dir is not None:
        imported = import_c0(Path(baseline_dir), out_dir, cases, manifest)
        print(f"C0: imported {imported} saved cases from {baseline_dir}", flush=True)
        report(out_dir, indices)

    for config in configs:
        profile = PROFILES[config]
        print(f"\n=== {config}: {profile} ===", flush=True)
        for position, case in enumerate(cases, 1):
            path = out_dir / config / "predictions" / f"{case.index:05d}.json"
            if path.exists():
                print(f"[{config} {position}/{count}] idx={case.index}: already saved", flush=True)
                continue
            print(f"[{config} {position}/{count}] idx={case.index}: generating...", flush=True)
            result = run_one(case, config=config, base_llm=base_llm,
                             state_dir=out_dir / config / "scenario_state",
                             generation_seed=generation_seed)
            atomic_json(path, result)
            print(f"  answer={result['prediction']} | gold={case.gold} | "
                  f"valid={result['valid']} | reason={result['invalid_reason']} | "
                  f"{result['seconds']:.1f}s | tokens={result['output_tokens']} | "
                  f"calls={result['llm_calls']}", flush=True)
            # Each task is persisted immediately; refresh aggregate reports
            # periodically to avoid excessive disk reads on long runs.
            if position % 10 == 0 or position == count:
                report(out_dir, indices)
    return report(out_dir, indices)


def main() -> int:
    parser = argparse.ArgumentParser(description="BICA ART/alphaNLI C0-C5 ablation suite")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--run-id", default="anli_ablations_pilot_01")
    parser.add_argument("--configs", default="C1,C2,C3,C4,C5",
                        help="Comma separated IDs; default C1..C5 to reuse existing C0")
    parser.add_argument("--reuse-c0", default=None,
                        help="Path to completed original C0 run (matching manifest)")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=2026)
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=2400)
    parser.add_argument("--results-dir", default="results/anli")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id can only include letters, digits, _ and -")
    configs = [s.strip().upper() for s in args.configs.split(",") if s.strip()]
    if len(set(configs)) != len(configs) or any(s not in PROFILES for s in configs):
        parser.error("--configs must contain unique values among C0,C1,C2,C3,C4,C5")
    if not configs and not args.reuse_c0:
        parser.error("--configs is empty and no --reuse-c0 is specified")

    print("Loading ART/alphaNLI validation...", flush=True)
    rows = load_validation()
    model = OllamaLLM(model=args.model, host=args.host,
                      temperature=args.temperature,
                      num_ctx=args.num_ctx, num_predict=args.num_predict)
    try:
        result = run_suite(
            rows, base_llm=model,
            output_dir=Path(args.results_dir) / args.run_id,
            baseline_dir=Path(args.reuse_c0) if args.reuse_c0 else None,
            configs=configs, count=args.count, offset=args.offset,
            sample_seed=args.sample_seed, generation_seed=args.generation_seed,
            model=args.model, temperature=args.temperature,
            num_ctx=args.num_ctx, num_predict=args.num_predict,
        )
    except KeyboardInterrupt:
        print("\nInterrupted; completed cases are saved. Rerun the same command.",
              file=sys.stderr, flush=True)
        return 130
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 2
    print("\nFinished. Current comparison:", flush=True)
    for config, s in result["by_config"].items():
        print(f"{config}: {s['correct']}/{s['processed']} "
              f"accuracy={s['accuracy_on_processed']} "
              f"invalid={s['invalid']} tokens={s['output_tokens']} "
              f"time={s['total_seconds']:.1f}s", flush=True)
    print("Saved to:", Path(args.results_dir) / args.run_id, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
