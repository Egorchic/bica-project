"""Belief-R sequential evaluation: direct baseline, agent, full agent."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from bica_agent import AgentConfig, OllamaLLM, ScenarioStore
from bica_agent.belief_revision import BeliefRevisionAgent
from bica_agent.client import IncompleteLLMResponse
from .belief_r import (
    DATASET,
    PROMPT_VERSION,
    SPLIT,
    BeliefRCase,
    cohort_hash,
    load_time_t1_cases,
    make_prompt,
    parse_answer,
    select_balanced_cases,
)

PROTOCOL = "BICA-BELIEF-R-v1"
PROFILES = {
    "B0": {"method": "direct", "k": 1, "evaluate": False,
           "self_refine": False, "history": False},
    "B1": {"method": "agent", "k": 3, "evaluate": True,
           "self_refine": False, "history": True},
    "B2": {"method": "agent", "k": 3, "evaluate": True,
           "self_refine": True, "history": True},
}


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    paths = {
        "runner": root / "experiments" / "run_belief_r_suite.py",
        "dataset_adapter": root / "experiments" / "belief_r.py",
        "belief_agent": root / "bica_agent" / "belief_revision.py",
        "store": root / "bica_agent" / "store.py",
        "ollama_client": root / "bica_agent" / "client.py",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def manifest_for(cases: list[BeliefRCase], *, model: str, temperature: float,
                 num_ctx: int, num_predict: int, sample_seed: int,
                 generation_seed: int) -> dict:
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "split": SPLIT,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "temperature": temperature,
        "think": True,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "sample_seed": sample_seed,
        "generation_seed": generation_seed,
        "count": len(cases),
        "indices": [c.index for c in cases],
        "dataset_ids": [c.dataset_id for c in cases],
        "cohort_sha256": cohort_hash(cases),
        "strata": {
            f"{relation}_{modus}": sum(
                c.relation == relation and c.modus == modus for c in cases
            )
            for relation in ("BU", "BM")
            for modus in ("ponens", "tollens")
        },
        "profiles": PROFILES,
        "source_sha256": code_fingerprints(),
    }


def ensure_manifest(path: Path, proposed: dict) -> None:
    if path.exists():
        if read_json(path) != proposed:
            raise ValueError(
                "Configuration, cohort, or source code changed. "
                "Use the same code/settings or create a new --run-id."
            )
    else:
        atomic_json(path, proposed)


class MeteredLLM:
    def __init__(self, base, *, case_seed_offset: int):
        self.base = base
        self.model = getattr(base, "model", "fake_llm")
        self.case_seed_offset = case_seed_offset
        self.records: list[dict] = []

    def ask(self, prompt: str, *, seed: int, stage: str):
        actual_seed = seed + self.case_seed_offset
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        def record(reply, *, incomplete: bool):
            self.records.append({
                "stage": stage,
                "seed": actual_seed,
                "prompt_sha256": digest,
                "response": reply.text,
                "seconds": reply.seconds,
                "output_tokens": reply.tokens,
                "done_reason": reply.done_reason,
                "incomplete": incomplete,
            })

        print(f"    [LLM] {stage}: start", flush=True)
        try:
            reply = self.base.ask(prompt, seed=actual_seed, stage=stage)
        except IncompleteLLMResponse as exc:
            record(exc.reply, incomplete=True)
            print(
                f"    [LLM] {stage}: {exc.reply.done_reason} | "
                f"{exc.reply.seconds:.1f}s | {exc.reply.tokens} tokens",
                flush=True,
            )
            raise
        record(reply, incomplete=False)
        print(
            f"    [LLM] {stage}: {reply.done_reason} | "
            f"{reply.seconds:.1f}s | {reply.tokens} tokens",
            flush=True,
        )
        return reply


def hid_to_answer(hid: str) -> str:
    mapping = {"H1": "a", "H2": "b", "H3": "c"}
    if hid not in mapping:
        raise ValueError(f"Unknown Belief-R hypothesis ID {hid!r}")
    return mapping[hid]


def _direct_step(case: BeliefRCase, *, step: str, llm: MeteredLLM) -> str:
    reply = llm.ask(
        make_prompt(case, step=step),
        seed=(0 if step == "time_t" else 10_000),
        stage=f"belief_direct_{step}",
    )
    answer = parse_answer(reply.text)
    if answer is None:
        raise ValueError("Missing or ambiguous ANSWER: A/B/C")
    return answer


def _agent_case(case: BeliefRCase, *, profile: dict, llm: MeteredLLM,
                state_dir: Path) -> tuple[str, str, dict]:
    store = ScenarioStore(state_dir)
    scenario = store.create_or_load(
        f"case_{case.index:05d}",
        [case.premise_1, case.premise_2],
        case.options,
    )
    agent = BeliefRevisionAgent(
        llm,
        store,
        AgentConfig(
            k=profile["k"],
            evaluate=profile["evaluate"],
            self_refine=profile["self_refine"],
            history=profile["history"],
        ),
    )

    # Resume safely even if the process stopped between the two timesteps.
    if scenario["rounds"]:
        round_t = scenario["rounds"][0]
    else:
        if len(scenario["observations"]) != 2:
            raise ValueError("Interrupted Belief-R state has P3 but no time_t round")
        round_t = agent.run_round(scenario)

    if case.premise_3 not in scenario["observations"]:
        store.add_observation(scenario, case.premise_3)
    round_t1 = agent.run_round(scenario)

    return (
        hid_to_answer(round_t["final"]),
        hid_to_answer(round_t1["final"]),
        {"time_t": round_t, "time_t1": round_t1},
    )


def run_one(case: BeliefRCase, *, config: str, base_llm,
            state_dir: Path, generation_seed: int) -> dict:
    profile = PROFILES[config]
    metered = MeteredLLM(
        base_llm,
        case_seed_offset=generation_seed + case.index,
    )
    pred_t = pred_t1 = None
    details: dict = {}
    invalid_reason = None
    try:
        if profile["method"] == "direct":
            pred_t = _direct_step(case, step="time_t", llm=metered)
            pred_t1 = _direct_step(case, step="time_t1", llm=metered)
        else:
            pred_t, pred_t1, details = _agent_case(
                case,
                profile=profile,
                llm=metered,
                state_dir=state_dir,
            )
    except IncompleteLLMResponse as exc:
        invalid_reason = (
            "truncated" if exc.reply.done_reason == "length"
            else "incomplete_response"
        )
    except ValueError as exc:
        invalid_reason = "invalid_format"
        details["error"] = str(exc)[:700]

    valid = invalid_reason is None and pred_t in {"a", "b", "c"} and pred_t1 in {"a", "b", "c"}
    correct_t = bool(valid and pred_t == case.gold_t)
    correct_t1 = bool(valid and pred_t1 == case.gold_t1)
    return {
        "index": case.index,
        "dataset_id": case.dataset_id,
        "atomic_idx": case.atomic_idx,
        "config": config,
        "modus": case.modus,
        "relation": case.relation,
        "prediction_t": pred_t if valid else None,
        "gold_t": case.gold_t,
        "correct_t": correct_t,
        "prediction_t1": pred_t1 if valid else None,
        "gold_t1": case.gold_t1,
        "correct_t1": correct_t1,
        "changed_prediction": bool(valid and pred_t != pred_t1),
        "expected_change": case.gold_t != case.gold_t1,
        "valid": valid,
        "invalid_reason": invalid_reason,
        "seconds": round(sum(c["seconds"] for c in metered.records), 2),
        "output_tokens": sum(c["output_tokens"] for c in metered.records),
        "llm_calls": len(metered.records),
        "calls": metered.records,
        "details": details,
    }


def load_results(out_dir: Path, config: str, indices: list[int]) -> list[dict]:
    return [
        read_json(path)
        for idx in indices
        if (path := out_dir / config / "predictions" / f"{idx:05d}.json").exists()
    ]


def _accuracy(items: list[dict], key: str) -> float | None:
    return round(sum(bool(x[key]) for x in items) / len(items), 6) if items else None


def summary_for(items: list[dict], n_selected: int) -> dict:
    valid = [x for x in items if x["valid"]]
    bu = [x for x in valid if x["relation"] == "BU"]
    bm = [x for x in valid if x["relation"] == "BM"]
    bu_acc = _accuracy(bu, "correct_t1")
    bm_acc = _accuracy(bm, "correct_t1")
    breu = (
        round((bu_acc + bm_acc) / 2, 6)
        if bu_acc is not None and bm_acc is not None else None
    )

    initial_ok = [x for x in valid if x["correct_t"]]
    bu_initial = [x for x in initial_ok if x["relation"] == "BU"]
    bm_initial = [x for x in initial_ok if x["relation"] == "BM"]

    return {
        "selected": n_selected,
        "processed": len(items),
        "complete": len(items) == n_selected,
        "valid": len(valid),
        "invalid": len(items) - len(valid),
        "truncated": sum(x.get("invalid_reason") == "truncated" for x in items),
        "basic_accuracy": _accuracy(valid, "correct_t"),
        "time_t1_accuracy": _accuracy(valid, "correct_t1"),
        "bu_count": len(bu),
        "bu_accuracy": bu_acc,
        "bm_count": len(bm),
        "bm_accuracy": bm_acc,
        "breu": breu,
        "initially_correct": len(initial_ok),
        "revision_accuracy_given_correct_t": _accuracy(initial_ok, "correct_t1"),
        "under_revision_rate_bu_given_correct_t": (
            round(sum(not x["changed_prediction"] for x in bu_initial) / len(bu_initial), 6)
            if bu_initial else None
        ),
        "over_revision_rate_bm_given_correct_t": (
            round(sum(x["changed_prediction"] for x in bm_initial) / len(bm_initial), 6)
            if bm_initial else None
        ),
        "total_seconds": round(sum(x["seconds"] for x in items), 2),
        "mean_seconds": round(sum(x["seconds"] for x in items) / len(items), 2) if items else None,
        "output_tokens": sum(x["output_tokens"] for x in items),
        "mean_output_tokens": round(sum(x["output_tokens"] for x in items) / len(items), 2) if items else None,
        "llm_calls": sum(x["llm_calls"] for x in items),
        "mean_llm_calls": round(sum(x["llm_calls"] for x in items) / len(items), 2) if items else None,
    }


def exact_mcnemar(a: list[dict], b: list[dict]) -> dict:
    map_a = {x["index"]: x for x in a}
    map_b = {x["index"]: x for x in b}
    if set(map_a) != set(map_b):
        raise ValueError("McNemar requires identical case IDs")
    a_only = sum(map_a[i]["correct_t1"] and not map_b[i]["correct_t1"] for i in map_a)
    b_only = sum(map_b[i]["correct_t1"] and not map_a[i]["correct_t1"] for i in map_a)
    discordant = a_only + b_only
    if discordant == 0:
        p = 1.0
    else:
        lowest = min(a_only, b_only)
        logs = [
            math.lgamma(discordant + 1) - math.lgamma(i + 1)
            - math.lgamma(discordant - i + 1) - discordant * math.log(2)
            for i in range(lowest + 1)
        ]
        largest = max(logs)
        p = min(1.0, 2 * math.exp(largest) * sum(math.exp(v - largest) for v in logs))
    return {
        "only_first_correct_t1": a_only,
        "only_second_correct_t1": b_only,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": round(p, 8),
    }


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
    for first, second in (("B0", "B1"), ("B0", "B2"), ("B1", "B2")):
        if first not in summaries or second not in summaries:
            continue
        if not summaries[first]["complete"] or not summaries[second]["complete"]:
            continue
        comparisons.append({
            "first": first,
            "second": second,
            **exact_mcnemar(all_results[first], all_results[second]),
        })

    result = {
        "protocol": PROTOCOL,
        "by_config": summaries,
        "paired_comparisons_t1": comparisons,
    }
    atomic_json(out_dir / "comparison.json", result)

    columns = [
        "config", "processed", "valid", "basic_accuracy", "time_t1_accuracy",
        "bu_accuracy", "bm_accuracy", "breu", "revision_accuracy_given_correct_t",
        "under_revision_rate_bu_given_correct_t", "over_revision_rate_bm_given_correct_t",
        "mean_seconds", "mean_output_tokens", "mean_llm_calls",
    ]
    with (out_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for config, s in summaries.items():
            writer.writerow({"config": config, **{k: s.get(k) for k in columns if k != "config"}})
    return result


def run_suite(cases: list[BeliefRCase], *, base_llm, output_dir: Path,
              configs: list[str], generation_seed: int = 2026) -> dict:
    if not configs or any(c not in PROFILES for c in configs):
        raise ValueError(f"Unknown/empty configs: {configs}")
    out_dir = Path(output_dir)
    indices = [case.index for case in cases]

    for config in configs:
        profile = PROFILES[config]
        print(f"\n=== {config}: {profile} ===", flush=True)
        for position, case in enumerate(cases, 1):
            path = out_dir / config / "predictions" / f"{case.index:05d}.json"
            if path.exists():
                print(f"[{config} {position}/{len(cases)}] idx={case.index}: already saved", flush=True)
                continue
            print(
                f"[{config} {position}/{len(cases)}] idx={case.index} "
                f"{case.relation}/{case.modus}: generating...",
                flush=True,
            )
            result = run_one(
                case,
                config=config,
                base_llm=base_llm,
                state_dir=out_dir / config / "scenario_state",
                generation_seed=generation_seed,
            )
            atomic_json(path, result)
            print(
                f"  t={result['prediction_t']}/{case.gold_t} | "
                f"t1={result['prediction_t1']}/{case.gold_t1} | "
                f"valid={result['valid']} | {result['seconds']:.1f}s | "
                f"tokens={result['output_tokens']} | calls={result['llm_calls']}",
                flush=True,
            )
            if position % 5 == 0 or position == len(cases):
                report(out_dir, indices)
    return report(out_dir, indices)


def main() -> int:
    parser = argparse.ArgumentParser(description="BICA Belief-R sequential suite")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--run-id", default="belief_r_main_50")
    parser.add_argument("--configs", default="B0,B1,B2")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=2026)
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=2400)
    parser.add_argument("--results-dir", default="results/belief_r")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id can only include letters, digits, _ and -")
    configs = [x.strip().upper() for x in args.configs.split(",") if x.strip()]
    if len(set(configs)) != len(configs) or any(x not in PROFILES for x in configs):
        parser.error("--configs must contain unique values among B0,B1,B2")

    print("Loading Belief-R...", flush=True)
    all_cases = load_time_t1_cases()
    cases = select_balanced_cases(all_cases, count=args.count, seed=args.sample_seed)
    out_dir = Path(args.results_dir) / args.run_id
    manifest = manifest_for(
        cases,
        model=args.model,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        sample_seed=args.sample_seed,
        generation_seed=args.generation_seed,
    )
    ensure_manifest(out_dir / "manifest.json", manifest)
    print(f"Selected {len(cases)} cases: {manifest['strata']}", flush=True)

    model = OllamaLLM(
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )
    try:
        result = run_suite(
            cases,
            base_llm=model,
            output_dir=out_dir,
            configs=configs,
            generation_seed=args.generation_seed,
        )
    except KeyboardInterrupt:
        print("\nInterrupted; completed cases are saved. Rerun the same command.", file=sys.stderr, flush=True)
        return 130
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 2

    print("\nFinished. Current comparison:", flush=True)
    for config, s in result["by_config"].items():
        print(
            f"{config}: basic={s['basic_accuracy']} "
            f"BU={s['bu_accuracy']} BM={s['bm_accuracy']} "
            f"BREU={s['breu']} invalid={s['invalid']}",
            flush=True,
        )
    print("Saved to:", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
