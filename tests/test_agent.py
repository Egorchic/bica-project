from bica_agent import AbductiveAgent, AgentConfig, ScenarioStore
from bica_agent.client import LLMReply


class FakeLLM:
    """Детерминированные ответы для проверки контроллера; не тестирует рассуждение LLM."""
    def ask(self, prompt, *, seed, stage):
        if stage == "generate":
            text = "H1: The person walked in rain.\nH2: A sprinkler splashed onto the person."
        elif stage.startswith("evaluate "):
            hid = stage.split()[-1]
            if "F3:" in prompt:
                if hid == "H1":
                    text = "F1: SUPPORT | Wetness fits rain.\nF2: CONTRADICT | No rain.\nF3: UNKNOWN | Notebook irrelevant."
                else:
                    text = "F1: SUPPORT | Wetness fits sprinkler.\nF2: UNKNOWN | No direct evidence of sprinkler.\nF3: UNKNOWN | Notebook irrelevant."
            elif "F2:" in prompt:
                if hid == "H1":
                    text = "F1: SUPPORT | Wetness fits rain.\nF2: CONTRADICT | No rain."
                else:
                    text = "F1: SUPPORT | Wetness fits sprinkler.\nF2: UNKNOWN | No evidence of sprinkler."
            else:
                text = "F1: SUPPORT | Consistent with wet clothes.\nF2: UNKNOWN | invented fact"
        elif stage.startswith("select") or stage == "tie breaker":
            text = "ANSWER: H2" if "F2: CONTRADICT" in prompt else "ANSWER: H1"
        elif stage == "self-refine critique":
            text = "Reasoning is adequate.\nREVIEW: KEEP"
        else:
            raise AssertionError(stage)
        return LLMReply(text=text, seconds=0.1, tokens=10)


def test_three_rounds_and_saved_history(tmp_path):
    store = ScenarioStore(tmp_path)
    agent = AbductiveAgent(FakeLLM(), store, AgentConfig(k=3))
    state = store.create_or_load("case", ["A person returned home wet."],
                                 ["Rain wet the person.", "A sprinkler wet the person."])
    r1 = agent.run_round(state)
    assert r1["final"] == "H1"
    assert len(r1["selection"]["votes"]) == 3
    store.add_observation(state, "There was no rain where the person went.")
    r2 = agent.run_round(state)
    assert r2["final"] == "H2"
    assert r2["changed_from_previous"] is True
    assert r2["evaluations"]["H2"]["F2"]["relation"] == "UNKNOWN"
    store.add_observation(state, "The person has a notebook.")
    r3 = agent.run_round(state)
    assert r3["final"] == "H2"
    assert r3["changed_from_previous"] is False
    assert len(state["rounds"]) == 3
    again = store.create_or_load("case", ["A person returned home wet."],
                                 ["Rain wet the person.", "A sprinkler wet the person."])
    assert len(again["rounds"]) == 3
    assert agent.run_round(again) == r3


def test_generation(tmp_path):
    store = ScenarioStore(tmp_path)
    agent = AbductiveAgent(FakeLLM(), store, AgentConfig(k=1, n_generate=2))
    state = store.create_or_load("free", ["A person returned home wet."])
    result = agent.run_round(state)
    assert len(result["hypotheses"]) == 2
    assert result["critique"]["verdict"] == "KEEP"


def test_direct_baseline_skips_evaluation_and_reflection(tmp_path):
    store = ScenarioStore(tmp_path)
    agent = AbductiveAgent(FakeLLM(), store,
                           AgentConfig(k=1, evaluate=False, self_refine=False))
    state = store.create_or_load("base", ["A person returned home wet."],
                                 ["Rain", "Sprinkler"])
    result = agent.run_round(state)
    assert result["evaluations"] == {}
    assert result["critique"]["verdict"] == "SKIP"
    assert len(result["calls"]) == 1


def test_scenario_isolation(tmp_path):
    store = ScenarioStore(tmp_path)
    one = store.create_or_load("one", ["Fact A"], ["H1", "H2"])
    two = store.create_or_load("two", ["Fact B"], ["H1", "H2"])
    store.add_observation(one, "New fact")
    assert two["observations"] == ["Fact B"]
