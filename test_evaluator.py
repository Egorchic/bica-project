from llm_client import LLMClient
from evaluator import evaluate_hypotheses


def main():
    llm = LLMClient()

    hypotheses = [
        "Человек промок, попав под садовый полив.",
        "Человек промок, попав под дождь.",
        "Человек промок, случайно пролив на себя воду."
    ]

    initial_observations = [
        "Человек вернулся домой в мокрой одежде."
    ]

    new_observation = (
        "В тот день во всех местах, где находился "
        "человек, атмосферных осадков не было."
    )

    cases = [
        ("Начальная оценка", initial_observations),
        (
            "После нового свидетельства",
            initial_observations + [new_observation]
        )
    ]

    expected = [
        [False, False, False],
        [False, True, False]
    ]

    for case_index, (name, observations) in enumerate(cases):
        print(f"\n=== {name} ===")

        result = evaluate_hypotheses(
            llm=llm,
            observations=observations,
            hypotheses=hypotheses
        )

        evaluations = sorted(
            result["evaluations"],
            key=lambda item: item["id"]
        )

        actual = []

        for item in evaluations:
            has_contradiction = bool(item["contradictions"])
            actual.append(has_contradiction)

            print(f"\nH{item['id']}: {hypotheses[item['id'] - 1]}")
            print("Противоречия:", item["contradictions"])

        print("\nОжидалось:", expected[case_index])
        print("Получено: ", actual)

        if actual == expected[case_index]:
            print("РЕЗУЛЬТАТ: PASS")
        else:
            print("РЕЗУЛЬТАТ: FAIL")

        print("Выбор:", result["best_id"])
        print("Обоснование:", result["reason"])


if __name__ == "__main__":
    main()
