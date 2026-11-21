import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:
    def __init__(self):
        # api_key = os.getenv("OPENROUTER_API_KEY")
        #
        # if not api_key:
        #     raise RuntimeError("Не найден OPENROUTER_API_KEY")
        #
        # self.model = os.getenv(
        #     "OPENROUTER_MODEL",
        #     "google/gemma-4-31b-it:free"
        # )
        #
        # self.client = OpenAI(
        #     api_key=api_key,
        #     base_url="https://openrouter.ai/api/v1",
        #     timeout=60,
        #     max_retries=0
        # )

        self.model = "deepseek-r1:8b"

        self.client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            timeout=180,
            max_retries=0
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        elapsed = time.perf_counter() - start
        print(f"[LLM] Время генерации: {elapsed:.2f} с")

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Модель вернула пустой ответ")

        return content.strip()
