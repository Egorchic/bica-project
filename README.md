# BICA — чистый экспериментальный Python-пакет

Перенос прототипа из Colab Notebook. **Не заменяет старые файлы в GitHub** и не претендует на доказанную эффективность. Проверка качества LLM ещё необходима.

## Состав

- `bica_agent/client.py`: адаптер Ollama для `qwen3:14b` с `think=True`.
- `bica_agent/store.py`: независимые сценарии, атомарное сохранение JSON, добавление наблюдений.
- `bica_agent/core.py`: генерация, оценка фактов, выбор с голосованием, рефлексия и один пересмотр.
- `examples/smoke.py`: демонстрация обоих режимов.
- `tests/test_agent.py`: быстрые тесты без модели и GPU.

Все текстовые промпты — на английском для англоязычных датасетов. Память **не переносится между сценариями**. Эталонные ответы в логике агента не используются.

## Google Colab

Убедись, что Ollama запущена и модель `qwen3:14b` скачана, как в ранее выполненных ячейках.
Загрузи и распакуй ZIP в `/content/bica_clean_project` или клонируй свой репозиторий после переноса файлов.

```python
%cd /content/bica_clean_project
!pip -q install -e '.[test]'
!pytest -q
!python -m examples.smoke --mode provided --k 1 --memory-dir /content/drive/MyDrive/bica_results/clean_agent
```

Для свободной генерации:

```python
!python -m examples.smoke --mode generate --k 1 --memory-dir /content/drive/MyDrive/bica_results/clean_agent
```

Чтобы запустить голосование, укажи `--k 3` и новый ID сценария в демонстрационном файле. Повторный запуск с уже обработанными наблюдениями не повторяет модельные вызовы.

## Эксперименты и абляции

```python
from bica_agent import AbductiveAgent, AgentConfig, OllamaLLM, ScenarioStore

agent = AbductiveAgent(
    OllamaLLM(model="qwen3:14b"),
    ScenarioStore("./results"),
    AgentConfig(k=3, evaluate=True, self_refine=True, history=True),
)
scenario = agent.store.create_or_load("example_01", ["..."], ["...", "..."])
round_data = agent.run_round(scenario)
```

Для ablation `k=1`, `evaluate=False`, `self_refine=False`, `history=False` по отдельности.
**Важно:** менять конфигурацию на *уже обработанном* ID нельзя для корректного сравнения; используйте отдельные ID и фиксируйте конфигурацию в итоговом экспериментальном протоколе.

## Известные ограничения

1. Семантическая оценка SUPPORT/CONTRADICT/UNKNOWN может ошибаться: промпт уточнён, но корректность на Qwen3 ещё не проверена.
2. Если процесс прервётся посередине раунда, завершённые раунды останутся сохранёнными, но вызовы незавершённого раунда выполнятся заново.
3. Голосование с нечётным k может дать ничью при трёх и более гипотезах; выполняется дополнительный запрос.
4. Генерация может породить причинно несостоятельную гипотезу. Нужен отдельный количественный анализ качества объяснений.
5. Адаптация механизмов Self-Consistency, Self-Refine и идеи эпизодической памяти Reflexion **не является полной репликацией оригинальных методов**.
