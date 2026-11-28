"""Post-hoc alphaNLG scoring with standard reference-based NLG metrics.

Required:
    pip install sacrebleu rouge-score

Optional BERTScore:
    pip install bert-score
    python -m experiments.score_anlg <run_dir> --bertscore

References are read only from already completed prediction JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config_items(run_dir: Path, config: str) -> list[dict]:
    pred_dir = run_dir / config / "predictions"
    if not pred_dir.exists():
        return []
    items = [read_json(p) for p in sorted(pred_dir.glob("*.json"))]
    return [x for x in items if x.get("valid") and x.get("prediction")]


def reference_streams(items: list[dict]) -> list[list[str]]:
    max_refs = max(len(x["references"]) for x in items)
    streams = [[] for _ in range(max_refs)]
    for item in items:
        refs = list(item["references"])
        if not refs:
            raise ValueError(f"No references for case {item.get('index')}")
        while len(refs) < max_refs:
            refs.append(refs[0])
        for i, ref in enumerate(refs):
            streams[i].append(ref)
    return streams


def score_config(items: list[dict], *, bertscore: bool) -> dict:
    if not items:
        return {}

    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError(
            "sacrebleu is required: pip install sacrebleu rouge-score"
        ) from exc
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError(
            "rouge-score is required: pip install sacrebleu rouge-score"
        ) from exc

    predictions = [x["prediction"] for x in items]
    refs_by_case = [list(x["references"]) for x in items]
    streams = reference_streams(items)

    bleu = sacrebleu.corpus_bleu(predictions, streams)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_values = []
    for pred, refs in zip(predictions, refs_by_case):
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref in refs]
        rouge_values.append(max(scores))

    result = {
        "n_scored": len(items),
        "sacrebleu": round(float(bleu.score), 6),
        "rougeL_f1_mean": round(statistics.fmean(rouge_values), 6),
        "bertscore_f1_mean": None,
    }

    if bertscore:
        try:
            from bert_score import score as bert_score
        except ImportError as exc:
            raise RuntimeError(
                "bert-score is required for --bertscore: pip install bert-score"
            ) from exc

        # GEM/ART normally exposes one public reference per row. If a row has
        # several references, use the first one consistently for BERTScore;
        # BLEU and ROUGE above use all available references.
        primary_refs = [refs[0] for refs in refs_by_case]
        _, _, f1 = bert_score(
            predictions,
            primary_refs,
            lang="en",
            verbose=True,
        )
        result["bertscore_f1_mean"] = round(float(f1.mean().item()), 6)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a completed alphaNLG run")
    parser.add_argument("run_dir")
    parser.add_argument("--bertscore", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "manifest.json").exists():
        print(f"Missing manifest: {run_dir / 'manifest.json'}", file=sys.stderr)
        return 2

    manifest = read_json(run_dir / "manifest.json")
    expected = manifest.get("count")
    results = {}
    for config in ("G0", "G1", "G2"):
        items = load_config_items(run_dir, config)
        if not items:
            continue
        metrics = score_config(items, bertscore=args.bertscore)
        metrics["selected"] = expected
        metrics["complete_valid_set"] = len(items) == expected
        results[config] = metrics
        (run_dir / config / "standard_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    output = {
        "run_dir": str(run_dir),
        "dataset": manifest.get("dataset"),
        "split": manifest.get("split"),
        "by_config": results,
    }
    (run_dir / "standard_metrics.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    columns = [
        "config",
        "n_scored",
        "selected",
        "complete_valid_set",
        "sacrebleu",
        "rougeL_f1_mean",
        "bertscore_f1_mean",
    ]
    with (run_dir / "standard_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for config, metrics in results.items():
            writer.writerow({"config": config, **metrics})

    for config, metrics in results.items():
        print(
            f"{config}: n={metrics['n_scored']} "
            f"SacreBLEU={metrics['sacrebleu']:.4f} "
            f"ROUGE-L={metrics['rougeL_f1_mean']:.4f} "
            f"BERTScore={metrics['bertscore_f1_mean']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
