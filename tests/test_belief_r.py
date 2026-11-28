from experiments.belief_r import BeliefRCase, extract_premises, select_balanced_cases


def row(modus="ponens", gold="c", dataset_id="x"):
    return {
        "questions": (
            "If P then Q\n"
            "P\n"
            "If R then Q\n"
            "What necessarily had to follow assuming that the above premises were true?\n"
            "(a) Q\n(b) not Q\n(c) maybe Q"
        ),
        "ground_truth": gold,
        "step": "time_t1",
        "modus": modus,
        "types_of_relation": "If-Event-Then-Event",
        "agreement_lv": 5,
        "atomic_idx": 1,
        "dataset_id": dataset_id,
        "a": "Q",
        "b": "not Q",
        "c": "maybe Q",
    }


def test_extract_premises():
    assert extract_premises(row()["questions"]) == ["If P then Q", "P", "If R then Q"]


def test_relation_from_gold_and_modus():
    assert BeliefRCase.from_row(1, row("ponens", "c")).relation == "BU"
    assert BeliefRCase.from_row(2, row("ponens", "a")).relation == "BM"
    assert BeliefRCase.from_row(3, row("tollens", "c")).relation == "BU"
    assert BeliefRCase.from_row(4, row("tollens", "b")).relation == "BM"


def test_balanced_50():
    cases = []
    index = 0
    for relation, modus, gold in (
        ("BU", "ponens", "c"),
        ("BU", "tollens", "c"),
        ("BM", "ponens", "a"),
        ("BM", "tollens", "b"),
    ):
        for i in range(30):
            cases.append(BeliefRCase.from_row(index, row(modus, gold, f"{relation}-{modus}-{i}")))
            index += 1
    selected = select_balanced_cases(cases, count=50, seed=42)
    assert len(selected) == 50
    assert sum(c.relation == "BU" for c in selected) == 25
    assert sum(c.relation == "BM" for c in selected) == 25
    assert sum(c.modus == "ponens" for c in selected) == 25
    assert sum(c.modus == "tollens" for c in selected) == 25
