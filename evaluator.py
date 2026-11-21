from typing import Literal
from pydantic import BaseModel

import json

from llm_client import LLMClient


class HypothesisEvaluation(BaseModel):
    id: int
    explains: list[str]
    contradictions: list[str]
    assumptions: list[str]


class EvaluationResult(BaseModel):
    evaluations: list[HypothesisEvaluation]
    best_id: int | None
    confidence: Literal[
        "низкая",
        "средняя",
        "высокая"
    ]
    reason: str


def evaluate_hypotheses(
    llm: LLMClient,
    observations: list[str],
    hypotheses: list[str]
) -> dict:
    system_prompt = """
    Ты проверяешь объяснительные гипотезы относительно известных фактов.

    Для каждой гипотезы независимо определи:

    - explains: какие наблюдения она способна объяснить.
    - contradictions: какие наблюдения логически несовместимы
      с этой гипотезой.
    - assumptions: какие дополнительные утверждения необходимы
      для гипотезы, но ещё не установлены.

    КРИТИЧЕСКИЕ ПРАВИЛА:

    1. Противоречие существует только тогда, когда наблюдение
       исключает событие, необходимое для гипотезы.

    2. Отсутствие подтверждения не является противоречием.

    3. Не расширяй смысл наблюдений.
       Например, "не было осадков" НЕ означает
       "не было воды" или "не было искусственного полива".

    4. Проверяй каждую гипотезу отдельно.
       Противоречие одной гипотезы не переносится на другие.

    5. Не придумывай новые наблюдения.

    6. В assumptions перечисляй конкретные непроверенные
       утверждения, а не фразы "неизвестно" или "неясно".

    7. Не считай гипотезу доказанной только потому,
       что она может объяснить наблюдение.

    8. Если данных недостаточно для обоснованного выбора,
       best_id должен быть null, confidence — "низкая".

    ПРИМЕР:

    Наблюдение: "В течение дня не было дождя".

    Гипотеза 1: "Человек промок под дождём".
    Противоречие: "В течение дня не было дождя".

    Гипотеза 2: "Человек промок под садовым поливом".
    Противоречий нет.

    Гипотеза 3: "Человек упал в бассейн".
    Противоречий нет.

    ФОРМАТ ОТВЕТА — ТОЛЬКО JSON:

    {
      "evaluations": [
        {
          "id": 1,
          "explains": ["Наблюдение"],
          "contradictions": [],
          "assumptions": ["Необходимое предположение"]
        }
      ],
      "best_id": null,
      "confidence": "низкая",
      "reason": "Краткое обоснование"
    }

    Оцени каждую переданную гипотезу.
    Допустимые confidence:
    "низкая", "средняя", "высокая".
    """

    data = {
        "observations": [
            {"id": i, "text": fact}
            for i, fact in enumerate(observations, 1)
        ],
        "hypotheses": [
            {"id": i, "text": hypothesis}
            for i, hypothesis in enumerate(hypotheses, 1)
        ]
    }

    raw = llm.generate(
        system_prompt=system_prompt,
        user_prompt=json.dumps(
            data, ensure_ascii=False, indent=2
        ),
        temperature=0.0,
        max_tokens=4500,
        think=True,
        response_schema=EvaluationResult.model_json_schema()
    )

    # Извлекаем JSON из ответа.
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

    evaluations = result.get("evaluations")

    if not isinstance(evaluations, list):
        raise ValueError("Нет списка evaluations")

    if len(evaluations) != len(hypotheses):
        raise ValueError("Оценены не все гипотезы")

    ids = [item.get("id") for item in evaluations
           if isinstance(item, dict)]

    if sorted(ids) != list(range(1, len(hypotheses) + 1)):
        raise ValueError("Некорректные ID гипотез")

    for item in evaluations:
        for field in ("explains", "contradictions", "assumptions"):
            value = item.get(field)
            if not isinstance(value, list) or not all(
                isinstance(x, str) for x in value
            ):
                raise ValueError(
                    f"Гипотеза H{item.get('id')}\n"
                    f"Некорректное поле: {field}\n"
                    f"Тип: {type(value).__name__}\n"
                    f"Значение: {value!r}\n\n"
                    f"Полный JSON:\n"
                    f"{json.dumps(result, ensure_ascii=False, indent=2)}"
                )

    best_id = result.get("best_id")

    if best_id is not None and best_id not in ids:
        raise ValueError("Некорректный best_id")

    if result.get("confidence") not in (
        "низкая", "средняя", "высокая"
    ):
        raise ValueError("Некорректная confidence")

    if not isinstance(result.get("reason"), str):
        raise ValueError("Отсутствует обоснование")

    return result
