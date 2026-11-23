"""C0: однократный прямой выбор Qwen3 на ART/αNLI.

Запуск: python -u -m experiments.run_anli_baseline --host http://IP:11434
Метки используются ТОЛЬКО при подсчёте метрик ПОСЛЕ генерации ответа.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys

from bica_agent.client import OllamaLLM
from .anli import (CONFIG, DATASET, PROMPT_VERSION, SPLIT, ANLICase,
                   cohort_hash, load_validation, make_direct_prompt,
                   parse_answer, select_indices)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_baseline(rows, *, llm, output_dir: Path, count: int = 30, offset: int = 0,
                 sample_seed: int = 42, generation_seed: int = 2026,
                 model: str = "qwen3:14b", temperature: float = 0.6,
                 num_ctx: int = 4096, num_predict: int = 2400) -> dict:
    indices = select_indices(len(rows), count=count, seed=sample_seed, offset=offset)
    cases = [ANLICase.from_row(i, rows[i]) for i in indices]
    manifest = {
        "protocol": "BICA-C0-ART-ANLI-v1", "dataset": DATASET,
        "dataset_config": CONFIG, "split": SPLIT,
        "prompt_version": PROMPT_VERSION, "model": model,
        "temperature": temperature, "num_ctx": num_ctx,
        "num_predict": num_predict, "think": True,
        "sample_seed": sample_seed, "generation_seed": generation_seed,
        "count": count, "offset": offset, "indices": indices,
        "cohort_sha256": cohort_hash(cases),
    }
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Параметры или данные отличаются от сохранённого запуска. Укажите другую --run-id.")
    else:
        save_json(manifest_path, manifest)

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for n, case in enumerate(cases, 1):
        result_path = predictions_dir / f"{case.index:05d}.json"
        if result_path.exists():
            print(f"[{n}/{count}] idx={case.index}: уже готово", flush=True)
            continue
        print(f"[{n}/{count}] idx={case.index}: запрос к LLM...", flush=True)
        prompt = make_direct_prompt(case)
        # Модель получает только prompt. Gold применяется уже после её ответа.
        reply = llm.ask(prompt, seed=generation_seed + case.index,
                        stage="anli_direct_baseline")
        prediction = parse_answer(reply.text)
        item = {
            "index": case.index, "prediction": prediction, "gold": case.gold,
            "correct": prediction == case.gold,
            "valid": prediction is not None,
            "response": reply.text, "seconds": reply.seconds,
            "output_tokens": reply.tokens, "calls": 1,
            "generation_seed": generation_seed + case.index,
            "done_reason": reply.done_reason,
        }
        save_json(result_path, item)
        print(f"  ответ={prediction} | правильный={case.gold} | "
              f"время={reply.seconds:.1f}с | токены={reply.tokens}", flush=True)
        save_json(output_dir / "summary.json", calculate_summary(output_dir, indices))

    summary = calculate_summary(output_dir, indices)
    save_json(output_dir / "summary.json", summary)
    return summary


def calculate_summary(output_dir: Path, indices: list[int]) -> dict:
    files = [Path(output_dir) / "predictions" / f"{idx:05d}.json" for idx in indices]
    results = [json.loads(f.read_text(encoding="utf-8")) for f in files if f.exists()]
    n = len(results)
    good = sum(bool(r["correct"]) for r in results)
    invalid = sum(not r["valid"] for r in results)
    return {
        "selected": len(indices), "processed": n,
        "correct": good, "invalid": invalid,
        "accuracy_on_processed": round(good / n, 6) if n else None,
        "total_seconds": round(sum(r["seconds"] for r in results), 2),
        "output_tokens": sum(r["output_tokens"] for r in results),
        "llm_calls": sum(r["calls"] for r in results),
        "complete": n == len(indices),
    }


def main():
    parser = argparse.ArgumentParser(description="ART/αNLI direct baseline (C0)")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--run-id", default="anli_c0_pilot_01")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=2400)
    parser.add_argument("--results-dir", default="results/anli")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        parser.error("--run-id может содержать только латинские буквы, цифры, _ и -")

    output_dir = Path(args.results_dir) / args.run_id
    print("Загрузка ART/αNLI validation...", flush=True)
    rows = load_validation()
    llm = OllamaLLM(model=args.model, host=args.host,
                    temperature=args.temperature,
                    num_predict=args.num_predict, num_ctx=args.num_ctx)
    try:
        summary = run_baseline(
            rows, llm=llm, output_dir=output_dir,
            count=args.count, offset=args.offset,
            sample_seed=args.sample_seed, generation_seed=args.generation_seed,
            model=args.model, temperature=args.temperature,
            num_ctx=args.num_ctx, num_predict=args.num_predict,
        )
    except KeyboardInterrupt:
        print("\nОстановлено пользователем. Завершённые примеры сохранены; повторите команду.",
              file=sys.stderr, flush=True)
        return 130
    print("Итоги:", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Директория:", output_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
