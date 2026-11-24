"""Offline tests for the ablation runner; no datasets download or Ollama required."""

import json

import pytest

from bica_agent.client import IncompleteLLMResponse, LLMReply
from experiments.anli import ANLICase, make_direct_prompt
from experiments.run_anli_suite import (
    PROFILES, exact_mcnemar, import_c0, manifest_for, report, run_one,
    run_suite, summary_for, wilson_interval, atomic_json,
)


def make_row(label=1, suffix=""):
    return {
        "observation_1": f"Alex entered a shop {suffix}.",
        "observation_2": f"Alex left with a book {suffix}.",
        "hypothesis_1": "Alex bought a book.",
        "hypothesis_2": "Alex looked at a poster.",
        "label": label,
    }


class FakeLLM:
    model = "qwen3:14b"

    def __init__(self, *, truncate_stage=None, revise=False):
        self.calls = []
        self.truncate_stage = truncate_stage
        self.revise = revise

    def ask(self, prompt, *, seed, stage):
        self.calls.append((prompt, seed, stage))
        if stage == self.truncate_stage:
            raise IncompleteLLMResponse(
                LLMReply(text="", seconds=2.5, tokens=2400, done_reason="length"), stage
            )
        if stage.startswith("anli_direct"):
            text = "ANSWER: H1"
        elif stage.startswith("evaluate"):
            text = "F1: UNKNOWN | Not probative.\nF2: SUPPORT | Plausible effect."
        elif stage.startswith("select") or stage == "tie breaker":
            text = ("ANSWER: H2" if "Reviewer note to consider:" in prompt
                    else "ANSWER: H1")
        elif stage == "self-refine critique":
            text = "Specific causal error found.\nREVIEW: REVISE" if self.revise else "REVIEW: KEEP"
        else:
            raise AssertionError(stage)
        return LLMReply(text=text, seconds=0.5, tokens=60, done_reason="stop")


def test_profiles_are_the_agreed_six_ablation_variants():
    assert list(PROFILES) == ["C0", "C1", "C2", "C3", "C4", "C5"]
    assert [PROFILES[x]["k"] for x in PROFILES] == [1, 3, 1, 1, 3, 3]
    assert [PROFILES[x]["self_refine"] for x in PROFILES] == [False, False, False, True, False, True]


def test_direct_baseline_prompt_seed_and_gold_isolation(tmp_path):
    case1 = ANLICase.from_row(44, make_row(label=1))
    case2 = ANLICase.from_row(44, make_row(label=2))
    assert make_direct_prompt(case1) == make_direct_prompt(case2)
    model = FakeLLM()
    result = run_one(case1, config="C0", base_llm=model,
                     state_dir=tmp_path, generation_seed=2026)
    assert result["prediction"] == 1
    assert result["correct"]
    assert result["llm_calls"] == 1
    assert model.calls[0] == (make_direct_prompt(case1), 2070, "anli_direct_vote_1")
    assert "Correct answer:" not in model.calls[0][0]
    assert "label" not in model.calls[0][0]


def test_c1_is_three_direct_votes_on_the_same_prompt(tmp_path):
    case = ANLICase.from_row(12, make_row())
    model = FakeLLM()
    result = run_one(case, config="C1", base_llm=model,
                     state_dir=tmp_path, generation_seed=2026)
    assert result["prediction"] == 1
    assert result["details"]["vote_counts"] == {1: 3}
    assert [call[1] for call in model.calls] == [2038, 2039, 2040]
    assert len(set(call[0] for call in model.calls)) == 1


def test_truncation_is_invalid_but_all_usage_is_accounted(tmp_path):
    case = ANLICase.from_row(3, make_row())
    model = FakeLLM(truncate_stage="anli_direct_vote_2")
    result = run_one(case, config="C1", base_llm=model,
                     state_dir=tmp_path, generation_seed=2026)
    assert not result["valid"]
    assert result["invalid_reason"] == "truncated"
    assert result["correct"] is False
    assert result["llm_calls"] == 2
    assert result["output_tokens"] == 2460
    assert result["seconds"] == 3.0


def test_agent_ablation_and_single_revision(tmp_path):
    case = ANLICase.from_row(9, make_row(label=2))
    model = FakeLLM(revise=True)
    result = run_one(case, config="C3", base_llm=model,
                     state_dir=tmp_path, generation_seed=2026)
    assert result["preliminary_prediction"] == 1
    assert result["prediction"] == 2
    assert result["correct"] is True
    assert result["details"]["critique"]["verdict"] == "REVISE"
    assert result["details"]["revision"] is not None
    assert result["llm_calls"] == 7  # 2 evaluation + select + critic + 2 evaluation + select
    assert result["calls"][0]["seed"] == 2026 + case.index + 1100
    assert (tmp_path / "case_00009.json").exists()
    s = summary_for([result], 1)
    assert s["reflection"]["wrong_to_right"] == 1
    assert s["reflection"]["right_to_wrong"] == 0


def test_agent_uses_two_observations_and_separate_scenario_memory(tmp_path):
    model = FakeLLM()
    row_a, row_b = make_row(suffix="A"), make_row(suffix="B")
    run_suite([row_a, row_b], base_llm=model, output_dir=tmp_path,
              configs=["C2", "C4"], count=2, sample_seed=12)
    saved_c2 = sorted((tmp_path / "C2" / "scenario_state").glob("*.json"))
    saved_c4 = sorted((tmp_path / "C4" / "scenario_state").glob("*.json"))
    assert len(saved_c2) == len(saved_c4) == 2
    for p in saved_c2 + saved_c4:
        state = json.loads(p.read_text(encoding="utf-8"))
        assert len(state["observations"]) == 2
        assert len(state["hypotheses"]) == 2
        assert len(state["rounds"]) == 1
        assert "gold" not in state
        assert state["rounds"][0]["configuration"]["history"] is False
    assert len([c for c in model.calls if c[2].startswith("select")]) == 8


def test_resume_and_manifest_lock(tmp_path):
    rows = [make_row(label=1, suffix="A"), make_row(label=2, suffix="B")]
    model = FakeLLM()
    kw = dict(rows=rows, base_llm=model, output_dir=tmp_path,
              configs=["C0", "C1"], count=2, sample_seed=9)
    result1 = run_suite(**kw)
    original_count = len(model.calls)
    result2 = run_suite(**kw)
    assert result1 == result2
    assert len(model.calls) == original_count
    assert result1["by_config"]["C0"]["complete"] is True
    with pytest.raises(ValueError, match="Конфигурация"):
        run_suite(**{**kw, "num_predict": 4096})


def test_invalid_agent_stage_does_not_stop_other_examples(tmp_path):
    model = FakeLLM(truncate_stage="self-refine critique")
    result = run_suite([make_row(label=2, suffix="x")], base_llm=model,
                       output_dir=tmp_path, configs=["C3"], count=1)
    s = result["by_config"]["C3"]
    assert s["processed"] == 1
    assert s["invalid"] == s["truncated"] == 1
    assert s["correct"] == 0
    assert s["llm_calls"] == 4


def test_import_baseline_checks_all_settings_and_reuses_results(tmp_path):
    case = ANLICase.from_row(0, make_row(label=2))
    manifest = manifest_for([case], model="qwen3:14b", temperature=0.6,
                            num_ctx=4096, num_predict=2400, sample_seed=42,
                            generation_seed=2026, offset=0)
    old = tmp_path / "old"
    current = tmp_path / "current"
    atomic_json(old / "manifest.json", {
        k: v for k, v in manifest.items() if k != "profiles" and k != "protocol"
    })
    atomic_json(old / "predictions" / "00000.json", {
        "index": 0, "gold": 2, "prediction": None, "correct": False,
        "valid": False, "invalid_reason": "truncated", "seconds": 7,
        "output_tokens": 2400, "calls": 1, "generation_seed": 2026,
        "done_reason": "length", "response": "",
    })
    assert import_c0(old, current, [case], manifest) == 1
    assert import_c0(old, current, [case], manifest) == 0
    imported = json.loads((current / "C0" / "predictions" / "00000.json").read_text())
    assert imported["invalid_reason"] == "truncated"
    assert imported["llm_calls"] == 1
    mismatched = {**manifest, "num_predict": 4096}
    with pytest.raises(ValueError, match="не совпадают"):
        import_c0(old, tmp_path / "wrong", [case], mismatched)


def test_report_contains_wilson_and_exact_paired_comparisons(tmp_path):
    s1 = {"index": 0, "config": "C0", "correct": False, "valid": False,
          "invalid_reason": "truncated", "seconds": 2, "output_tokens": 50,
          "llm_calls": 1}
    s2 = {"index": 0, "config": "C2", "correct": True, "valid": True,
          "invalid_reason": None, "seconds": 4, "output_tokens": 100,
          "llm_calls": 3}
    atomic_json(tmp_path / "C0" / "predictions" / "00000.json", s1)
    atomic_json(tmp_path / "C2" / "predictions" / "00000.json", s2)
    out = report(tmp_path, [0])
    assert out["by_config"]["C0"]["truncated"] == 1
    assert out["by_config"]["C2"]["accuracy_on_processed"] == 1
    assert out["paired_comparisons"][0]["discordant"] == 1
    assert out["paired_comparisons"][0]["mcnemar_exact_two_sided_p"] == 1.0
    assert (tmp_path / "comparison.csv").exists()
    assert wilson_interval(0, 0) is None
    assert exact_mcnemar([s1], [s2])["only_second_correct"] == 1
