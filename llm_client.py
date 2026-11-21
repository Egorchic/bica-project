
import time

from ollama import Client


class LLMClient:
    def __init__(self):
        self.model = "deepseek-r1:8b"

        self.client = Client(
            host="http://localhost:11434",
            timeout=600
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        think: bool = True,
        response_schema: dict | None = None
    ) -> str:

        start = time.perf_counter()

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            think=think,
            format=response_schema if response_schema is not None else "json",
            stream=False,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 8192
            }
        )

        elapsed = time.perf_counter() - start

        content = (response.message.content or "").strip()
        thinking = response.message.thinking or ""

        print(f"[LLM] Модель: {self.model}")
        print(f"[LLM] Время: {elapsed:.2f} с")
        print(f"[LLM] Завершение: {response.done_reason}")
        print(f"[LLM] Выходных токенов: {response.eval_count}")
        print(f"[LLM] Символов рассуждений: {len(thinking)}")
        print(f"[LLM] Символов ответа: {len(content)}")

        if response.done_reason == "length":
            raise ValueError(
                "Модель исчерпала бюджет генерации. "
                f"Лимит: {max_tokens} токенов."
            )

        if not content:
            raise ValueError(
                "Модель не сформировала окончательный ответ"
            )

        return content
