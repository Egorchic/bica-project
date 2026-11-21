
import json

from llm_client import LLMClient


def check_contradictions(
    llm: LLMClient,
    observations: list[str],
    hypothesis: str
) -> dict:

    system_prompt = """
Ты проверяешь логическую совместимость одной гипотезы
с набором наблюдений.

Противоречие существует, если гипотеза и наблюдение
не могут быть истинными одновременно.

Правила:
1. Проверяй только переданную гипотезу.
2. Используй исключительно предоставленные наблюдения.
3. Не придумывай дополнительные факты.
4. Не расширяй смысл наблюдений.
5. Отсутствие подтверждения не является противоречием.
6. Если гипотеза может быть истинна одновременно
   с наблюдением, противоречия нет.
7. Не оценивай правдоподобие и не выбирай
   лучшее объяснение.

Верни только JSON следующего формата:

{
  "contradiction_ids": [],
  "reason": "Краткое объяснение"
}

В contradiction_ids указывай только числовые ID
реально предоставленных наблюдений,
которые противоречат гипотезе.

Если противоречий нет, возвращай пустой список.
"""

    facts = [
        {"id": i, "text": text}
        for i, text in enumerate(observations, 1)
    ]

    user_prompt = json.dumps(
        {
            "observations": facts,
            "hypothesis": hypothesis
        },
        ensure_ascii=False,
        indent=2
    )

    raw = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=250
    )

    start = raw.find("{")

    if start == -1:
        raise ValueError(f"JSON не найден:\n{raw}")

    try:
        result, _ = json.JSONDecoder().raw_decode(
            raw[start:]
        )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Некорректный JSON:\n{raw}"
        ) from e

    if not isinstance(result, dict):
        raise ValueError("Ожидался JSON-объект")

    ids = result.get("contradiction_ids")

    if not isinstance(ids, list):
        raise ValueError("Нет списка contradiction_ids")

    if any(
        type(i) is not int or not 1 <= i <= len(observations)
        for i in ids
    ):
        raise ValueError(f"Некорректные ID: {ids}")

    if len(ids) != len(set(ids)):
        raise ValueError("Повторяющиеся ID наблюдений")

    reason = result.get("reason")

    if not isinstance(reason, str):
        raise ValueError("Некорректное поле reason")

    return {
        "contradiction_ids": ids,
        "contradictions": [
            observations[i - 1] for i in ids
        ],
        "reason": reason
    }
