# 10. Указывай причинный механизм, который непосредственно приводит к наблюдению. Не пропускай необходимые промежуточные события.

import json
from json import JSONDecodeError

from llm_client import LLMClient


def generate_hypotheses(
    llm: LLMClient,
    observations: list[str],
    n: int = 3
) -> list[str]:

    if not observations:
        raise ValueError("Список наблюдений пуст")

    system_prompt = """
Ты агент, выполняющий абдуктивное рассуждение.

Твоя задача — предложить несколько различных
правдоподобных объяснений наблюдаемых фактов.

Правила:
1. Генерируй конкурирующие гипотезы.
2. Каждая гипотеза должна объяснять наблюдения.
3. Не повторяй одну идею разными словами.
4. Не представляй предположения как установленные факты.
5. Пока не выбирай лучшую гипотезу.
6. Возвращай только корректный JSON.
7. Каждая гипотеза должна описывать одну конкретную причину. Не объединяй разные объяснения через "или" и не перечисляй несколько сценариев внутри одной гипотезы.
7. Гипотезы должны описывать принципиально разные причинные механизмы.
8. Не считай разными гипотезами переформулировки одного объяснения. Например, "попал под дождь" и "промок во время ливня" — одна причина.
9. Каждая гипотеза должна описывать один конкретный сценарий без перечисления альтернатив.
10. Указывай причинный механизм, который непосредственно приводит к наблюдению. Не пропускай необходимые промежуточные события.

Формат ответа:
{
    "hypotheses": [
        "Первая гипотеза",
        "Вторая гипотеза",
        "Третья гипотеза"
    ]
}
"""

    facts = "\n".join(
        f"{i}. {observation}"
        for i, observation in enumerate(observations, 1)
    )

    user_prompt = f"""
Наблюдения:
{facts}

Сгенерируй ровно {n} различных гипотез,
объясняющих эти наблюдения.

Верни JSON с полем "hypotheses",
содержащим массив из {n} строк.
"""

    raw_response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=500
    )

    # Пробуем разобрать ответ модели как JSON.
    # Если есть текст до JSON, ищем начало объекта.
    try:
        data = json.loads(raw_response)
    except JSONDecodeError:
        start = raw_response.find("{")

        if start == -1:
            raise ValueError(
                f"Ответ не содержит JSON:\n{raw_response}"
            )

        try:
            data, _ = json.JSONDecoder().raw_decode(
                raw_response[start:]
            )
        except JSONDecodeError as e:
            raise ValueError(
                f"Некорректный JSON:\n{raw_response}"
            ) from e

    if not isinstance(data, dict):
        raise ValueError("Ожидался JSON-объект")

    hypotheses = data.get("hypotheses")

    if not isinstance(hypotheses, list):
        raise ValueError("Отсутствует список hypotheses")

    if len(hypotheses) != n:
        raise ValueError(
            f"Ожидалось {n} гипотез, "
            f"получено {len(hypotheses)}"
        )

    if not all(
        isinstance(h, str) and h.strip()
        for h in hypotheses
    ):
        raise ValueError("Некорректные гипотезы")

    hypotheses = [h.strip() for h in hypotheses]

    if len(set(h.lower() for h in hypotheses)) != n:
        raise ValueError("Обнаружены одинаковые гипотезы")

    return hypotheses
