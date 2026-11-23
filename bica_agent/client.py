"""Изолированный адаптер Ollama. Бизнес-логика не зависит от библиотеки Ollama."""
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class LLMReply:
    text: str
    seconds: float = 0.0
    tokens: int = 0
    done_reason: str = "stop"


class OllamaLLM:
    def __init__(self, model: str = "qwen3:14b", host: str = "http://127.0.0.1:11434",
                 temperature: float = 0.6, num_predict: int = 2400, num_ctx: int = 4096):
        from ollama import Client
        self.client = Client(host=host, timeout=900)
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx

    def ask(self, prompt: str, *, seed: int, stage: str) -> LLMReply:
        start = perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            think=True,
            stream=False,
            keep_alive="30m",
            options={
                "temperature": self.temperature,
                "seed": seed,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        )
        text = (response.message.content or "").strip()
        reason = response.done_reason or ""
        if reason != "stop" or not text:
            raise RuntimeError(f"Незавершённый ответ на этапе {stage}: {reason}; {text[:200]!r}")
        return LLMReply(
            text=text,
            seconds=round(perf_counter() - start, 2),
            tokens=response.eval_count or 0,
            done_reason=reason,
        )
