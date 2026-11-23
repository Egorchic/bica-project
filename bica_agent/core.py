"""Одна архитектура, два режима: готовые гипотезы и свободная генерация.

Механизмы: проверка фактов, Self-Consistency, самокритика по мотивам Self-Refine,
эпизодическая память в пределах сценария. Нет доступа к правильным меткам.
"""
from collections import Counter
from dataclasses import dataclass, asdict
import re

from .client import LLMReply
from .store import ScenarioStore


@dataclass(frozen=True)
class AgentConfig:
    k: int = 3
    n_generate: int = 3
    evaluate: bool = True
    self_refine: bool = True
    history: bool = True

    def __post_init__(self):
        if self.k < 1 or self.k % 2 == 0:
            raise ValueError("k должно быть положительным нечётным числом")
        if self.n_generate < 2:
            raise ValueError("n_generate должно быть >= 2")


def _tag(text: str, name: str, choices: set[str]) -> str:
    matches = []
    for line in text.splitlines():
        clean = line.strip().replace("**", "").strip("` ")
        match = re.fullmatch(rf"{re.escape(name)}\s*:\s*([A-Za-z0-9]+)\.?", clean, re.I)
        if match:
            matches.append(match.group(1).upper())
    if not matches or matches[-1] not in choices:
        raise ValueError(f"Нет корректной строки {name}: ... в ответе: {text[:500]!r}")
    return matches[-1]


def _facts(observations):
    return "\n".join(f"F{i}: {fact}" for i, fact in enumerate(observations, 1))


def _hypotheses(hypotheses):
    return "\n".join(f"{key}: {value}" for key, value in hypotheses.items())


def _evaluations(evaluations):
    lines = []
    for hyp_id, facts in evaluations.items():
        lines.append(f"{hyp_id}:")
        for fid, value in facts.items():
            lines.append(f"  {fid}: {value['relation']} | {value['reason']}")
    return "\n".join(lines)


class AbductiveAgent:
    def __init__(self, llm, store: ScenarioStore, config: AgentConfig | None = None):
        self.llm = llm
        self.store = store
        self.config = config or AgentConfig()

    def _ask(self, prompt: str, *, seed: int, stage: str, trace: list) -> str:
        reply: LLMReply = self.llm.ask(prompt, seed=seed, stage=stage)
        trace.append({
            "stage": stage, "seconds": reply.seconds, "tokens": reply.tokens,
            "done_reason": reply.done_reason, "response": reply.text,
        })
        return reply.text

    def _generate(self, scenario: dict, round_no: int, trace: list):
        n = self.config.n_generate
        template = "\n".join(f"H{i}: causal explanation" for i in range(1, n+1))
        prompt = f"""Generate exactly {n} distinct causal hypotheses for these observations:
{_facts(scenario['observations'])}

Each hypothesis must have a different causal mechanism and explain how the
observations arose. Ensure that event order and location make sense. Do not
invent established facts. Do not rank them yet.
Output exactly these lines, replacing the explanation text:
{template}"""
        text = self._ask(prompt, seed=100+round_no, stage="generate", trace=trace)
        found = {}
        for line in text.splitlines():
            m = re.fullmatch(r"\s*(H\d+)\s*:\s*(.+)", line.replace("**", ""), re.I)
            if m:
                key = m.group(1).upper()
                if key in found:
                    raise ValueError("Повтор ID в генерации")
                found[key] = m.group(2).strip()
        expected = [f"H{i}" for i in range(1, n+1)]
        if list(found) != expected:
            raise ValueError(f"Некорректные гипотезы: {text}")
        scenario["hypotheses"] = found
        self.store.save(scenario)

    def _evaluate(self, scenario: dict, round_no: int, trace: list, *, correction="", seed_base=1000):
        observations = scenario["observations"]
        expected = [f"F{i}" for i in range(1, len(observations)+1)]
        example_format = "\n".join(f"{f}: RELATION | brief reason" for f in expected)
        results = {}
        for i, (hid, hyp) in enumerate(scenario["hypotheses"].items()):
            prompt = f"""Evaluate ONE abductive explanation against EACH fact independently.
Facts:
{_facts(observations)}
Hypothesis {hid}: {hyp}
{('Reviewer note: '+correction) if correction else ''}

Use these labels:
SUPPORT = the fact is positive evidence for THIS specific causal explanation;
CONTRADICT = this hypothesis and the fact cannot both be true;
UNKNOWN = the fact neither supports this particular cause nor contradicts it.

CRITICAL: Compatibility alone is NOT SUPPORT. Refuting a DIFFERENT hypothesis
is NOT, by itself, positive evidence for THIS causal explanation. Never turn
absence of support into contradiction. Respect all time, location and universal
constraints. Do not invent missing facts; consider whether the proposed cause
can actually precede and produce the observation.

Output ONLY {len(observations)} lines, using precisely the IDs below:
{example_format}
Replace RELATION with SUPPORT, CONTRADICT, or UNKNOWN."""
            text = self._ask(prompt, seed=seed_base+round_no*100+i,
                             stage=f"evaluate {hid}", trace=trace)
            parsed = {}
            for line in text.splitlines():
                cleaned = line.strip().replace("**", "").strip("` ")
                match = re.fullmatch(
                    r"F(\d+)\s*:\s*(SUPPORT|CONTRADICT|UNKNOWN)(?:\s*\|\s*(.*))?",
                    cleaned, re.I,
                )
                if match:
                    fid = f"F{int(match.group(1))}"
                    if fid in parsed:
                        raise ValueError(f"Повтор {fid} в оценке {hid}")
                    parsed[fid] = {"relation": match.group(2).upper(),
                                   "reason": (match.group(3) or "").strip()}
            missing = set(expected) - set(parsed)
            if missing:
                raise ValueError(f"Отсутствуют оценки {sorted(missing)} у {hid}: {text}")
            # Лишние F-метки игнорируются, но не попадают в память как реальные факты.
            results[hid] = {f: parsed[f] for f in expected}
        return results

    def _select(self, scenario: dict, evaluations: dict, round_no: int,
                trace: list, *, reviewer_note="", seed_base=2000):
        note = f"\nReviewer note to consider: {reviewer_note}" if reviewer_note else ""
        eval_text = (_evaluations(evaluations) if evaluations else
                     "No separate evidence evaluation was performed.")
        prompt = f"""Choose the most plausible causal explanation linking the facts.
Facts:
{_facts(scenario['observations'])}
Hypotheses:
{_hypotheses(scenario['hypotheses'])}
Evidence assessments (may be incorrect; reason critically):
{eval_text}{note}

Compare causal coherence, explanatory coverage, contradictions and unsupported
assumptions. A hypothesis may be possible without being well supported.
Do not treat a previous selection or the absence of a competing cause as new
observed evidence. End with exactly one line: ANSWER: H1
(or another available hypothesis ID)."""
        available = set(scenario["hypotheses"])
        votes = []
        for i in range(self.config.k):
            text = self._ask(prompt, seed=seed_base+round_no*100+i,
                             stage=f"select {i+1}/{self.config.k}", trace=trace)
            votes.append({"answer": _tag(text, "ANSWER", available), "text": text})
        counts = Counter(v["answer"] for v in votes)
        maximum = max(counts.values())
        winners = [h for h, count in counts.items() if count == maximum]
        tie_breaker = None
        if len(winners) == 1:
            chosen = winners[0]
        else:
            tied_prompt = prompt + "\nVotes tied. Select ONLY among: " + ", ".join(winners)
            text = self._ask(tied_prompt, seed=seed_base+round_no*100+99,
                             stage="tie breaker", trace=trace)
            chosen = _tag(text, "ANSWER", set(winners))
            tie_breaker = text
        return {"chosen": chosen, "counts": dict(counts), "votes": votes,
                "tie_breaker": tie_breaker}

    def _reflect(self, scenario: dict, evaluations: dict, selection: dict,
                 round_no: int, trace: list):
        history = ""
        if self.config.history and scenario["rounds"]:
            prev = scenario["rounds"][-1]
            history = (f"\nEarlier decision in this scenario: {prev['final']}.\n"
                       f"Earlier reviewer note: {prev['critique']['note']}.\n"
                       "These are historical decisions, NOT facts.\n")
        prompt = f"""Critically review this abductive decision without access to any answer key.
Facts:
{_facts(scenario['observations'])}
Hypotheses:
{_hypotheses(scenario['hypotheses'])}
Evaluations:
{_evaluations(evaluations) if evaluations else 'Not separately evaluated.'}
Preliminary choice: {selection['chosen']}
Votes: {selection['counts']}{history}

Identify concrete reasoning mistakes, such as false claims of support, violated
causal timelines, omitted facts, invented exceptions or actual contradictions.
Merely noting that several causes are plausible or that evidence is insufficient
is NOT necessarily an error and does not by itself justify re-evaluating.
Write a concise critique, then end with exactly one line:
REVIEW: KEEP
or
REVIEW: REVISE
Choose REVISE only if a concrete error could change the evaluation or choice."""
        text = self._ask(prompt, seed=3000+round_no, stage="self-refine critique", trace=trace)
        verdict = _tag(text, "REVIEW", {"KEEP", "REVISE"})
        note = "\n".join(line for line in text.splitlines()
                         if not re.match(r"\s*REVIEW\s*:", line, re.I)).strip()
        return {"verdict": verdict, "note": note, "text": text}

    def run_round(self, scenario: dict) -> dict:
        if (scenario["rounds"] and
            scenario["rounds"][-1]["observations"] == scenario["observations"]):
            return scenario["rounds"][-1]
        round_no = len(scenario["rounds"]) + 1
        trace = []
        if not scenario["hypotheses"]:
            self._generate(scenario, round_no, trace)
        evaluations = (self._evaluate(scenario, round_no, trace) if self.config.evaluate else {})
        selection = self._select(scenario, evaluations, round_no, trace)
        critique = (self._reflect(scenario, evaluations, selection, round_no, trace)
                    if self.config.self_refine else {"verdict": "SKIP", "note": "", "text": ""})
        revision = None
        final = selection["chosen"]
        if critique["verdict"] == "REVISE":
            new_evals = (self._evaluate(scenario, round_no, trace,
                                        correction=critique["note"], seed_base=4000)
                         if self.config.evaluate else {})
            new_sel = self._select(scenario, new_evals, round_no, trace,
                                   reviewer_note=critique["note"], seed_base=5000)
            revision = {"evaluations": new_evals, "selection": new_sel}
            final = new_sel["chosen"]
        previous = scenario["rounds"][-1]["final"] if scenario["rounds"] else None
        item = {
            "round": round_no,
            "observations": list(scenario["observations"]),
            "hypotheses": dict(scenario["hypotheses"]),
            "evaluations": evaluations,
            "selection": selection,
            "critique": critique,
            "revision": revision,
            "final": final,
            "previous_final": previous,
            "changed_from_previous": None if previous is None else final != previous,
            "seconds": round(sum(x["seconds"] for x in trace), 2),
            "tokens": sum(x["tokens"] for x in trace),
            "calls": trace,
            "configuration": asdict(self.config),
            "model": getattr(self.llm, "model", "test_or_custom_backend"),
        }
        scenario["rounds"].append(item)
        self.store.save(scenario)
        return item
