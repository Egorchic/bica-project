
from llm_client import LLMClient
from contradiction_checker import check_contradictions


def main():
    llm = LLMClient()

    hypotheses = [
        "Человек промок, попав под садовый полив.",
        "Человек промок, попав под дождь.",
        "Человек промок, случайно пролив на себя воду."
    ]

    initial_facts = [
        "Человек вернулся домой в мокрой одежде."
    ]

    new_fact = (
        "В тот день во всех местах, где находился "
        "человек, атмосферных осадков не было."
    )

    tests = [
        {
            "name": "Начальная оценка",
            "observations": initial_facts,
            "expected": [False, False, False]
        },
        {
            "name": "После нового свидетельства",
            "observations": initial_facts + [new_fact],
            "expected": [False, True, False]
        }
    ]

    for test in tests:
        print(f"\n=== {test['name']} ===")

        actual = []

        for i, hypothesis in enumerate(hypotheses, 1):
            result = check_contradictions(
                llm=llm,
                observations=test["observations"],
                hypothesis=hypothesis
            )

            has_conflict = bool(
                result["contradiction_ids"]
            )

            actual.append(has_conflict)

            print(f"\nH{i}: {hypothesis}")
            print("Противоречия:", result["contradictions"])
            print("Обоснование:", result["reason"])

        print("\nОжидалось:", test["expected"])
        print("Получено: ", actual)

        status = (
            "PASS"
            if actual == test["expected"]
            else "FAIL"
        )

        print("РЕЗУЛЬТАТ:", status)


if __name__ == "__main__":
    main()
