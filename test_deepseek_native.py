
import json
import time
from urllib.request import Request, urlopen


payload = {
    "model": "deepseek-r1:8b",
    "messages": [
        {
            "role": "user",
            "content": (
                "Назови три различные причины мокрой одежды. "
                "Верни только JSON с полем hypotheses, "
                "содержащим массив из трёх строк."
            )
        }
    ],
    "think": False,
    "format": "json",
    "stream": False,
    "options": {
        "temperature": 0.3,
        "num_predict": 1000,
        "num_ctx": 4096
    }
}

request = Request(
    "http://localhost:11434/api/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

start = time.perf_counter()

with urlopen(request, timeout=180) as response:
    result = json.load(response)

elapsed = time.perf_counter() - start

message = result.get("message", {})
content = message.get("content") or ""
thinking = message.get("thinking") or ""

print(f"Время: {elapsed:.2f} с")
print("Причина завершения:", result.get("done_reason"))
print("Выходных токенов:", result.get("eval_count"))
print("Символов в thinking:", len(thinking))
print("Символов в content:", len(content))

print("\n=== ОТВЕТ ===")
print(content)

if content:
    try:
        data = json.loads(content)
        print("\nJSON корректен:", data)
    except json.JSONDecodeError as e:
        print("\nОшибка JSON:", e)
