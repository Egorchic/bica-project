from openai import RateLimitError
from openai import OpenAIError

from llm_client import LLMClient
from hypotheses import generate_hypotheses
from evaluator import evaluate_hypotheses


def main():
    llm = LLMClient()

    observations = [
        "Человек вернулся домой в мокрой одежде."
    ]

    try:
        hypotheses = generate_hypotheses(
            llm, observations, n=3
        )

        print("Сгенерированные гипотезы:")

        for i, h in enumerate(hypotheses, 1):
            print(f"H{i}: {h}")

        result = evaluate_hypotheses(
            llm, observations, hypotheses
        )

        print("\nПроверка гипотез:")

        for item in result["evaluations"]:
            print(f"\nH{item['id']}")
            print("Объясняет:", item["explains"])
            print("Противоречия:", item["contradictions"])
            print("Предположения:", item["assumptions"])

        print("\nПредварительный выбор:",
              result["best_id"])
        print("Уверенность:", result["confidence"])
        print("Обоснование:", result["reason"])

        # Новое свидетельство
        new_fact = (
            "Установлено, что в тот день во всех местах, "
            "где находился человек, осадков не было."
        )

        updated_observations = observations + [new_fact]

        print("\n=== ПОСТУПИЛО НОВОЕ СВИДЕТЕЛЬСТВО ===")
        print(new_fact)

        # Проверяем прежние гипотезы с учётом нового факта
        updated_result = evaluate_hypotheses(
            llm=llm,
            observations=updated_observations,
            hypotheses=hypotheses
        )

        print("\n=== ПОВТОРНАЯ ОЦЕНКА ===")

        for item in updated_result["evaluations"]:
            print(f"\nH{item['id']}: {hypotheses[item['id'] - 1]}")
            print("Объясняет:", item["explains"])
            print("Противоречия:", item["contradictions"])
            print("Предположения:", item["assumptions"])

        print("\nПредыдущий выбор:", result["best_id"])
        print("Новый выбор:", updated_result["best_id"])
        print("Уверенность:", updated_result["confidence"])
        print("Обоснование:", updated_result["reason"])


    except RateLimitError as e:
        print("\nОшибка 429")
        print("Детали:", e.body)

    except (OpenAIError, ValueError) as e:
        print(f"\nОшибка: {e}")


if __name__ == "__main__":
    main()
