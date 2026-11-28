import json

from bica_agent.musr import _answer, _review
from experiments.musr import (
    DOMAINS,
    MuSRCase,
    cases_from_rows,
    select_cases,
)


def test_answer_parser():
    assert _answer("Reasoning\nANSWER: 2", 3) == 1


def test_answer_parser_rejects_multiple_answers():
    try:
        _answer("ANSWER: 1\nANSWER: 2", 3)
    except ValueError:
        pass
    else:
        raise AssertionError("multiple answers must be invalid")


def test_review_parser():
    assert _review("Concrete issue.\nREVIEW: REVISE") == "REVISE"


def test_cases_from_rows_accepts_json_questions():
    rows = [{
        "context": "Story.",
        "questions": json.dumps([{
            "question": "Who?",
            "answer": 1,
            "choices": ["A", "B"],
        }]),
    }]
    cases = cases_from_rows("murder_mysteries", rows)
    assert len(cases) == 1
    assert cases[0].gold == 1
    assert cases[0].choices == ("A", "B")


def _synthetic(domain, n, choices):
    return [
        MuSRCase(
            uid=f"{domain}:{i}:0",
            domain=domain,
            source_index=i,
            question_index=0,
            context=f"story {i}",
            question="q",
            choices=tuple(str(x) for x in range(choices)),
            gold=0,
        )
        for i in range(n)
    ]


def test_first_50_are_17_17_16():
    pools = {
        "murder_mysteries": _synthetic("murder_mysteries", 100, 2),
        "object_placements": _synthetic("object_placements", 100, 5),
        "team_allocation": _synthetic("team_allocation", 100, 3),
    }
    selected = select_cases(pools, count=50, offset=0, seed=42)
    counts = {d: sum(c.domain == d for c in selected) for d in DOMAINS}
    assert counts == {
        "murder_mysteries": 17,
        "object_placements": 17,
        "team_allocation": 16,
    }



def test_cases_from_rows_accepts_hf_flat_csv_schema():
    rows = [{
        "narrative": "Detective story.",
        "question": "Who is the most likely murderer?",
        "choices": "['Alice', 'Bob']",
        "answer_index": 1,
        "answer_choice": "Bob",
    }]
    cases = cases_from_rows("murder_mysteries", rows)
    assert len(cases) == 1
    assert cases[0].context == "Detective story."
    assert cases[0].choices == ("Alice", "Bob")
    assert cases[0].gold == 1


def test_object_placements_allows_variable_choice_count():
    rows = [
        {
            "narrative": "Story A.",
            "question": "Where?",
            "choices": "['desk', 'drawer', 'shelf']",
            "answer_index": 1,
            "answer_choice": "drawer",
        },
        {
            "narrative": "Story B.",
            "question": "Where?",
            "choices": "['a', 'b', 'c', 'd', 'e']",
            "answer_index": 4,
            "answer_choice": "e",
        },
    ]
    cases = cases_from_rows("object_placements", rows)
    assert [len(c.choices) for c in cases] == [3, 5]
    assert [c.gold for c in cases] == [1, 4]
