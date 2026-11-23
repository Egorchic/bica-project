import json

import pytest

from bica_agent.client import LLMReply, IncompleteLLMResponse
from experiments.anli import ANLICase, make_direct_prompt, parse_answer, select_indices
from experiments.run_anli_baseline import calculate_summary, run_baseline


def row(label=1):
    return {
        "observation_1": "A person heard thunder.",
        "observation_2": "The pavement was wet.",
        "hypothesis_1": "It rained.",
        "hypothesis_2": "Someone painted the street.",
        "label": label,
    }


class FakeLLM:
    def __init__(self, response="ANSWER: H1"):
        self.prompts = []
        self.response = response

    def ask(self, prompt, *, seed, stage):
        assert stage == "anli_direct_baseline"
        self.prompts.append((prompt, seed))
        return LLMReply(text=self.response, seconds=0.4, tokens=20, done_reason="stop")


def test_indices_are_reproducible_and_nonoverlapping():
    a = select_indices(1532, count=30, offset=0, seed=42)
    b = select_indices(1532, count=300, offset=30, seed=42)
    assert len(a) == 30
    assert len(b) == 300
    assert not set(a) & set(b)
    assert a == select_indices(1532, count=30, offset=0, seed=42)
    with pytest.raises(ValueError):
        select_indices(10, count=11, offset=0, seed=42)


def test_prompt_never_includes_label_field():
    c1 = ANLICase.from_row(7, row(1))
    c2 = ANLICase.from_row(7, row(2))
    assert make_direct_prompt(c1) == make_direct_prompt(c2)
    assert "Observation 1" in make_direct_prompt(c1)
    assert "H2:" in make_direct_prompt(c1)


def test_answer_parser_is_strict():
    assert parse_answer("Reasoning...\nANSWER: H2") == 2
    assert parse_answer("**ANSWER: H1**") == 1
    assert parse_answer("ANSWER: H3") is None
    assert parse_answer("ANSWER: H1\nANSWER: H2") is None
    assert parse_answer("H1") is None


def test_run_saves_and_resumes_without_new_calls(tmp_path):
    rows = [row(1), row(2), row(1)]
    llm = FakeLLM()
    kwargs = dict(rows=rows, llm=llm, output_dir=tmp_path,
                  count=2, offset=0, sample_seed=0)
    summary = run_baseline(**kwargs)
    assert summary["processed"] == 2
    assert summary["llm_calls"] == 2
    assert len(llm.prompts) == 2
    assert run_baseline(**kwargs) == summary
    assert len(llm.prompts) == 2
    saved = list((tmp_path / "predictions").glob("*.json"))
    assert len(saved) == 2
    assert all(json.loads(p.read_text(encoding="utf-8"))["prediction"] == 1 for p in saved)
    assert calculate_summary(tmp_path, json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["indices"]) == summary


def test_different_configuration_cannot_reuse_results(tmp_path):
    rows = [row(1), row(2), row(1)]
    llm = FakeLLM()
    run_baseline(rows, llm=llm, output_dir=tmp_path, count=1)
    with pytest.raises(ValueError, match="Параметры или данные"):
        run_baseline(rows, llm=llm, output_dir=tmp_path, count=1, temperature=0.1)


def test_invalid_answers_are_not_silently_dropped(tmp_path):
    llm = FakeLLM("ANSWER: H9")
    result = run_baseline([row(1)], llm=llm, output_dir=tmp_path, count=1)
    assert result["processed"] == 1
    assert result["invalid"] == 1
    assert result["correct"] == 0
    assert result["accuracy_on_processed"] == 0


class TruncatedLLM:
    def ask(self, prompt, *, seed, stage):
        raise IncompleteLLMResponse(
            LLMReply(text="", seconds=77.2, tokens=2400, done_reason="length"),
            stage,
        )


def test_truncation_is_saved_as_invalid_with_costs(tmp_path):
    summary = run_baseline([row(1)], llm=TruncatedLLM(), output_dir=tmp_path, count=1)
    assert summary["complete"] is True
    assert summary["correct"] == 0
    assert summary["invalid"] == 1
    assert summary["truncated"] == 1
    assert summary["total_seconds"] == 77.2
    assert summary["output_tokens"] == 2400
    path = next((tmp_path / "predictions").glob("*.json"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["prediction"] is None
    assert saved["invalid_reason"] == "truncated"
    assert saved["done_reason"] == "length"
    assert saved["calls"] == 1
    # A repeated command must never silently replace recorded model failures.
    assert run_baseline([row(1)], llm=TruncatedLLM(), output_dir=tmp_path, count=1) == summary


class NetworkFailureLLM:
    def ask(self, prompt, *, seed, stage):
        raise ConnectionError("network unavailable")


def test_connection_errors_are_not_counted_as_model_mistakes(tmp_path):
    with pytest.raises(ConnectionError, match="network unavailable"):
        run_baseline([row(1)], llm=NetworkFailureLLM(), output_dir=tmp_path, count=1)
    assert not list((tmp_path / "predictions").glob("*.json"))
