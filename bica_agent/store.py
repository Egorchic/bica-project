"""Эпизодическая память отдельного сценария, без переноса между задачами."""
import json
import re
from pathlib import Path


class ScenarioStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, scenario_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", scenario_id):
            raise ValueError("Идентификатор сценария должен содержать только буквы, цифры, _ или -")
        return self.directory / f"{scenario_id}.json"

    def create_or_load(self, scenario_id: str, observations: list[str],
                       hypotheses: list[str] | None = None) -> dict:
        path = self._path(scenario_id)
        if path.exists():
            scenario = json.loads(path.read_text(encoding="utf-8"))
            if scenario["observations"][:len(observations)] != observations:
                raise ValueError("Сценарий с этим ID уже существует и содержит другие исходные факты")
            if hypotheses is not None:
                expected = {f"H{i}": h for i, h in enumerate(hypotheses, 1)}
                if scenario["mode"] != "provided" or scenario["hypotheses"] != expected:
                    raise ValueError("Сценарий с этим ID содержит другие гипотезы")
            elif scenario["mode"] != "generate":
                raise ValueError("Сценарий с этим ID имеет другой режим")
            return scenario

        if not observations or any(not v.strip() for v in observations):
            raise ValueError("Нужен непустой список наблюдений")
        if hypotheses is not None and (len(hypotheses) < 2 or any(not h.strip() for h in hypotheses)):
            raise ValueError("В режиме выбора нужно минимум две непустые гипотезы")
        state = {
            "id": scenario_id,
            "mode": "generate" if hypotheses is None else "provided",
            "observations": list(observations),
            "hypotheses": {} if hypotheses is None else {f"H{i}": h for i, h in enumerate(hypotheses, 1)},
            "rounds": [],
        }
        self.save(state)
        return state

    def save(self, scenario: dict):
        path = self._path(scenario["id"])
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def add_observation(self, scenario: dict, fact: str):
        if not fact.strip():
            raise ValueError("Факт не должен быть пустым")
        if fact not in scenario["observations"]:
            scenario["observations"].append(fact)
            self.save(scenario)
