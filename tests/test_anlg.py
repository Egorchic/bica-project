from experiments.anlg import (
    ANLGCase,
    cohort_hash,
    make_direct_prompt,
    parse_bridge,
    select_indices,
)


def test_case_accepts_target_and_references():
    a = ANLGCase.from_row(
        7,
        {
            "gem_id": "x",
            "observation_1": "A happened.",
            "observation_2": "C happened.",
            "target": "B happened.",
            "references": ["B happened."],
        },
    )
    assert a.index == 7
    assert a.references == ("B happened.",)

    b = ANLGCase.from_row(
        8,
        {
            "observation_1": "A happened.",
            "observation_2": "C happened.",
            "label": "B happened.",
        },
    )
    assert b.references == ("B happened.",)


def test_sampling_is_deterministic_and_nonoverlapping():
    first = select_indices(1000, count=100, offset=0, seed=42)
    again = select_indices(1000, count=100, offset=0, seed=42)
    second = select_indices(1000, count=100, offset=100, seed=42)
    assert first == again
    assert len(first) == len(set(first)) == 100
    assert set(first).isdisjoint(second)


def test_bridge_parser_requires_exactly_one_final_marker():
    assert parse_bridge("Reasoning\nBRIDGE: He called a mechanic.") == (
        "He called a mechanic."
    )
    assert parse_bridge("No marker") is None
    assert parse_bridge("BRIDGE: A\nBRIDGE: B") is None


def test_prompt_contains_no_reference():
    case = ANLGCase(
        1,
        "g1",
        "Kim missed the bus.",
        "Kim arrived at work.",
        ("Kim called a taxi.",),
    )
    prompt = make_direct_prompt(case)
    assert "Kim missed the bus." in prompt
    assert "Kim arrived at work." in prompt
    assert "Kim called a taxi." not in prompt


def test_cohort_hash_changes_with_reference():
    a = ANLGCase(1, "g", "A", "C", ("B1",))
    b = ANLGCase(1, "g", "A", "C", ("B2",))
    assert cohort_hash([a]) != cohort_hash([b])
