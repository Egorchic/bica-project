"""GEM/ART alphaNLG generation suite: G0-G2.

G0: direct one-shot bridge generation.
G1: BICA free generation (3 candidates) + evidence evaluation + 3 selection votes.
G2: G1 + self-refine critique and, when needed, re-evaluation/re-selection.

Human references are attached only after all model calls have finished.
Each completed case is persisted atomically and the run is resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from bica_agent import AgentConfig, OllamaLLM, ScenarioStore
from bica_agent.anlg import AlphaNLGAgent
from bica_agent.client import IncompleteLLMResponse
from .anlg import (
    ANLGCase, DATASET, PROMPT_VERSION, SPLIT, cohort_hash, load_validation,
    make_direct_prompt, parse_bridge, select_indices,
)

PROTOCOL = "BICA-ART-ANLG-G0-G2-v1"
PROFILES = {
    "G0": {
        "method": "direct",
        "k": 1,
        "n_generate": 1,
        "evaluate": False,
        "self_refine": False,
    },
    "G1": {
        "method": "agent",
        "k": 3,
        "n_generate": 3,
        "evaluate": True,
        "self_refine": False,
    },
    "G2": {
        "method": "agent",
        "k": 3,
        "n_generate": 3,
        "evaluate": True,
        "self_refine": True,
    },
}


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    paths = {
        "suite": root / "experiments" / "run_anlg_suite.py",
        "dataset_adapter": root / "experiments" / "anlg.py",
        "task_agent": root / "bica_agent" / "anlg.py",
        "agent_core": root / "bica_agent" / "core.py",
        "ollama_client": root / "bica_agent" / "client.py",
        "scenario_store": root / "bica_agent" / "store.py",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def manifest_for(
    cases: list[ANLGCase],
    *,
    model: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    sample_seed: int,
    generation_seed: int,
    offset: int,
) -> dict:
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
        "offset": offset,
        "indices": [c.index for c in cases],
        "cohort_sha256": cohort_hash(cases),
        "profiles": PROFILES,
        "source_sha256": code_fingerprints(),
    }


def ensure_manifest(path: Path, proposed: dict) -> None:
    if path.exists():
        if read_json(path) != proposed:
            raise ValueError(
                "Configuration, cohort, or source code changed. "
                "Use a new --run-id; do not mix incompatible experiments."
            )
    else:
        atomic_json(path, proposed)


class MeteredLLM:
    def __init__(self, base, *, agent_seed_offset: int):
        self.base = base
        self.model = getattr(base, "model", "fake_llm")
        self.agent_seed_offset = agent_seed_offset
        self.records: list[dict] = []

    def ask(self, prompt: str, *, seed: int, stage: str):
        actual_seed = (
            seed
            if stage.startswith("anlg_direct")
            else seed + self.agent_seed_offset
        )
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        def record(reply, *, incomplete: bool):
            self.records.append(
                {
                    "stage": stage,
                    "seed": actual_seed,
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


class InvalidOutput(ValueError):
    pass


def run_one(
    case: ANLGCase,
    *,
    config: str,
    base_llm,
    state_dir: Path,
    generation_seed: int,
) -> dict:
    profile = PROFILES[config]
    metered = MeteredLLM(
        base_llm,
        agent_seed_offset=generation_seed + case.index,
    )
    prediction = None
    details: dict = {}
    invalid_reason = None

    try:
        if profile["method"] == "direct":
            reply = metered.ask(
                make_direct_prompt(case),
                seed=generation_seed + case.index,
                stage="anlg_direct",
            )
            prediction = parse_bridge(reply.text)
            if prediction is None:
                raise InvalidOutput("Missing or ambiguous BRIDGE: line")
            details = {"raw_generation": reply.text}
        else:
            store = ScenarioStore(state_dir)
            state = store.create_or_load(
                f"case_{case.index:05d}",
                [case.observation_1, case.observation_2],
                hypotheses=None,
            )
            agent = AlphaNLGAgent(
                metered,
                store,
                AgentConfig(
                    k=profile["k"],
                    n_generate=profile["n_generate"],
                    evaluate=profile["evaluate"],
                    self_refine=profile["self_refine"],
                    history=False,
                ),
            )
            round_result = agent.run_round(state)
            final_id = round_result["final"]
            if final_id not in state["hypotheses"]:
                raise InvalidOutput(f"Unknown final hypothesis ID: {final_id!r}")
            prediction = state["hypotheses"][final_id].strip()
            if not prediction:
                raise InvalidOutput("Selected bridge is empty")
            details = {
                "candidates": dict(state["hypotheses"]),
                "final_id": final_id,
                "evaluations": round_result["evaluations"],
                "selection": round_result["selection"],
                "critique": round_result["critique"],
                "revision": round_result["revision"],
            }

            # If state was saved but the result JSON was not, recover costs from
            # the persisted completed round instead of silently reporting zero.
            if not metered.records and round_result.get("calls"):
                metered.records = [
                    {
                        "stage": r["stage"],
                        "seed": None,
                        "prompt_sha256": None,
                        "response": r["response"],
                        "seconds": r["seconds"],
                        "output_tokens": r["tokens"],
                        "done_reason": r["done_reason"],
                        "incomplete": False,
                        "recovered_from_scenario": True,
                    }
                    for r in round_result["calls"]
                ]

    except IncompleteLLMResponse as exc:
        invalid_reason = (
            "truncated"
            if exc.reply.done_reason == "length"
            else "incomplete_response"
        )
    except ValueError as exc:
        invalid_reason = "invalid_format"
        details["error"] = str(exc)[:1000]

    valid = invalid_reason is None and isinstance(prediction, str) and bool(prediction.strip())

    # IMPORTANT: references are attached only here, after every LLM call above.
    return {
        "index": case.index,
        "gem_id": case.gem_id,
        "config": config,
        "observation_1": case.observation_1,
        "observation_2": case.observation_2,
        "prediction": prediction.strip() if valid else None,
        "references": list(case.references),
        "valid": valid,
        "invalid_reason": invalid_reason,
        "seconds": round(sum(c["seconds"] for c in metered.records), 2),
        "output_tokens": sum(c["output_tokens"] for c in metered.records),
        "llm_calls": len(metered.records),
        "calls": metered.records,
        "details": details,
    }


def load_results(out_dir: Path, config: str, indices: list[int]) -> list[dict]:
    items = []
    for idx in indices:
        path = out_dir / config / "predictions" / f"{idx:05d}.json"
        if path.exists():
            items.append(read_json(path))
    return items


def summary_for(items: list[dict], n_selected: int) -> dict:
    n = len(items)
    valid = sum(bool(x["valid"]) for x in items)
    seconds = sum(x["seconds"] for x in items)
    output_tokens = sum(x["output_tokens"] for x in items)
    calls = sum(x["llm_calls"] for x in items)
    return {
        "selected": n_selected,
        "processed": n,
        "complete": n == n_selected,
        "valid": valid,
        "invalid": n - valid,
        "truncated": sum(
            x.get("invalid_reason") == "truncated" for x in items
        ),
        "total_seconds": round(seconds, 2),
        "mean_seconds": round(seconds / n, 2) if n else None,
        "output_tokens": output_tokens,
        "mean_output_tokens": round(output_tokens / n, 2) if n else None,
        "llm_calls": calls,
        "mean_llm_calls": round(calls / n, 2) if n else None,
        "note": (
            "Text quality metrics are intentionally computed separately with "
            "experiments.score_anlg so generation does not depend on evaluator "
            "packages or expose references to the model."
        ),
    }


def report(out_dir: Path, indices: list[int]) -> dict:
    summaries = {}
    for config in PROFILES:
        items = load_results(out_dir, config, indices)
        if not items:
            continue
        summaries[config] = summary_for(items, len(indices))
        atomic_json(out_dir / config / "summary.json", summaries[config])

    results = {
        "protocol": PROTOCOL,
        "by_config": summaries,
    }
    atomic_json(out_dir / "comparison.json", results)

    csv_path = out_dir / "comparison.csv"
    columns = [
        "config",
        "processed",
        "valid",
        "invalid",
        "truncated",
        "mean_seconds",
        "mean_output_tokens",
        "mean_llm_calls",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for config, s in summaries.items():
            writer.writerow(
                {"config": config, **{k: s[k] for k in columns if k != "config"}}
            )
    return results


def run_suite(
    rows,
    *,
    base_llm,
    output_dir: Path,
    configs: list[str],
    count: int = 100,
    offset: int = 0,
    sample_seed: int = 42,
    generation_seed: int = 2026,
    model: str = "qwen3:14b",
    temperature: float = 0.6,
    num_ctx: int = 4096,
    num_predict: int = 2400,
) -> dict:
    if not configs:
        raise ValueError("No configurations to run")
    if any(c not in PROFILES for c in configs):
        raise ValueError(f"Unknown configs: {configs}")

    indices = select_indices(
        len(rows),
        count=count,
        offset=offset,
        seed=sample_seed,
    )
    cases = [ANLGCase.from_row(i, rows[i]) for i in indices]
    out_dir = Path(output_dir)

    manifest = manifest_for(
        cases,
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        sample_seed=sample_seed,
        generation_seed=generation_seed,
        offset=offset,
    )
    ensure_manifest(out_dir / "manifest.json", manifest)

    for config in configs:
        profile = PROFILES[config]
        print(f"\n=== {config}: {profile} ===", flush=True)
        for position, case in enumerate(cases, 1):
            path = out_dir / config / "predictions" / f"{case.index:05d}.json"
            if path.exists():
                print(
                    f"[{config} {position}/{count}] idx={case.index}: already saved",
                    flush=True,
                )
                continue

            print(
                f"[{config} {position}/{count}] idx={case.index}: generating...",
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
            preview = (result["prediction"] or "<invalid>").replace("\n", " ")[:120]
            print(
                f"  valid={result['valid']} | reason={result['invalid_reason']} | "
                f"{result['seconds']:.1f}s | tokens={result['output_tokens']} | "
                f"calls={result['llm_calls']} | {preview}",
                flush=True,
            )
            if position % 10 == 0 or position == count:
                report(out_dir, indices)

    return report(out_dir, indices)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BICA GEM/ART alphaNLG G0-G2 generation suite"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--run-id", default="anlg_main_100")
    parser.add_argument("--configs", default="G0,G1,G2")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=2026)
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=2400)
    parser.add_argument("--results-dir", default="results/anlg")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id can only include letters, digits, _ and -")

    configs = [s.strip().upper() for s in args.configs.split(",") if s.strip()]
    if len(set(configs)) != len(configs) or any(s not in PROFILES for s in configs):
        parser.error("--configs must contain unique values among G0,G1,G2")

    print("Loading GEM/ART validation...", flush=True)
    rows = load_validation()
    model = OllamaLLM(
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )

    try:
        result = run_suite(
            rows,
            base_llm=model,
            output_dir=Path(args.results_dir) / args.run_id,
            configs=configs,
            count=args.count,
            offset=args.offset,
            sample_seed=args.sample_seed,
            generation_seed=args.generation_seed,
            model=args.model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted; completed cases are saved. Rerun the same command.",
            file=sys.stderr,
            flush=True,
        )
        return 130
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 2

    print("\nFinished. Current generation summary:", flush=True)
    for config, s in result["by_config"].items():
        print(
            f"{config}: processed={s['processed']} valid={s['valid']} "
            f"invalid={s['invalid']} tokens={s['output_tokens']} "
            f"time={s['total_seconds']:.1f}s",
            flush=True,
        )
    print("Saved to:", Path(args.results_dir) / args.run_id, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
