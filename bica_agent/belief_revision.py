"""Belief-revision task adapter using BICA evaluator/selector/reflection mechanics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import re

from .core import AgentConfig
from .client import LLMReply
from .store import ScenarioStore


def _tag(text: str, name: str, choices: set[str]) -> str:
    matches = []
    for line in text.splitlines():
        clean = line.strip().replace("**", "").strip("` ")
        match = re.fullmatch(
            rf"{re.escape(name)}\s*:\s*([A-Za-z0-9]+)\.?", clean, re.I
        )
        if match:
            matches.append(match.group(1).upper())
    if not matches or matches[-1] not in choices:
        raise ValueError(f"Missing valid {name}: ... line in {text[:500]!r}")
    return matches[-1]


def _premises(observations: list[str]) -> str:
    return "\n".join(f"P{i}: {fact}" for i, fact in enumerate(observations, 1))


def _options(hypotheses: dict[str, str]) -> str:
    names = ["A", "B", "C"]
    return "\n".join(
        f"{hid} (option {names[i]}): {text}"
        for i, (hid, text) in enumerate(hypotheses.items())
    )


def _evaluation_text(evaluation: dict) -> str:
    lines = [f"NEW_INFO: {evaluation['new_info']} | {evaluation['new_info_reason']}"]
    for hid, item in evaluation["options"].items():
        lines.append(f"{hid}: {item['status']} | {item['reason']}")
    return "\n".join(lines)


class BeliefRevisionAgent:
    """Task-specific reasoning shell; no gold labels are exposed to the model."""

    def __init__(self, llm, store: ScenarioStore, config: AgentConfig):
        self.llm = llm
        self.store = store
        self.config = config

    def _ask(self, prompt: str, *, seed: int, stage: str, trace: list) -> str:
        reply: LLMReply = self.llm.ask(prompt, seed=seed, stage=stage)
        trace.append({
            "stage": stage,
            "seconds": reply.seconds,
            "tokens": reply.tokens,
            "done_reason": reply.done_reason,
            "response": reply.text,
        })
        return reply.text

    def _evaluate(self, scenario: dict, round_no: int, trace: list,
                  *, correction: str = "", seed_base: int = 1000) -> dict:
        has_new = len(scenario["observations"]) >= 3
        relation_instruction = (
            "P3 is newly introduced. Classify its role as ADDITIONAL if its antecedent "
            "is a commonsense prerequisite that makes the earlier conclusion depend on "
            "another requirement; ALTERNATIVE if it is an independent route to the same "
            "consequent; or NONE if that distinction is not applicable."
            if has_new else
            "There is no newly introduced P3 yet, so NEW_INFO must be NONE."
        )
        correction_text = f"\nReviewer note: {correction}" if correction else ""
        prompt = f"""Analyze this Belief-R reasoning state without access to an answer key.
Premises:
{_premises(scenario['observations'])}
Candidate conclusions:
{_options(scenario['hypotheses'])}

Belief-R uses defeasible commonsense reasoning in addition to basic modus
ponens/tollens. A later conditional can reveal an additional necessary condition
for the earlier inference, or merely an alternative sufficient route.
{relation_instruction}{correction_text}

For each candidate conclusion use:
SUPPORTED = the current premises make it the best necessary conclusion;
CONTRADICTED = it conflicts with the current reasoning state;
UNDETERMINED = the premises leave it genuinely open.

Output exactly four lines:
NEW_INFO: ADDITIONAL|ALTERNATIVE|NONE | brief reason
H1: SUPPORTED|CONTRADICTED|UNDETERMINED | brief reason
H2: SUPPORTED|CONTRADICTED|UNDETERMINED | brief reason
H3: SUPPORTED|CONTRADICTED|UNDETERMINED | brief reason"""
        text = self._ask(
            prompt,
            seed=seed_base + round_no * 100,
            stage="belief evaluate",
            trace=trace,
        )

        parsed: dict[str, dict] = {}
        new_info = None
        new_info_reason = ""
        for line in text.splitlines():
            clean = line.strip().replace("**", "").strip("` ")
            m_rel = re.fullmatch(
                r"NEW_INFO\s*:\s*(ADDITIONAL|ALTERNATIVE|NONE)(?:\s*\|\s*(.*))?",
                clean,
                re.I,
            )
            if m_rel:
                new_info = m_rel.group(1).upper()
                new_info_reason = (m_rel.group(2) or "").strip()
                continue
            m_opt = re.fullmatch(
                r"(H[123])\s*:\s*(SUPPORTED|CONTRADICTED|UNDETERMINED)"
                r"(?:\s*\|\s*(.*))?",
                clean,
                re.I,
            )
            if m_opt:
                hid = m_opt.group(1).upper()
                if hid in parsed:
                    raise ValueError(f"Duplicate evaluation for {hid}")
                parsed[hid] = {
                    "status": m_opt.group(2).upper(),
                    "reason": (m_opt.group(3) or "").strip(),
                }

        if new_info is None or set(parsed) != {"H1", "H2", "H3"}:
            raise ValueError(f"Invalid Belief-R evaluation format: {text[:700]!r}")
        if not has_new and new_info != "NONE":
            raise ValueError("Evaluator claimed new information before P3 was introduced")
        return {
            "new_info": new_info,
            "new_info_reason": new_info_reason,
            "options": parsed,
        }

    def _select(self, scenario: dict, evaluation: dict, round_no: int,
                trace: list, *, reviewer_note: str = "",
                seed_base: int = 2000) -> dict:
        previous = scenario["rounds"][-1]["final"] if scenario["rounds"] else None
        history = ""
        if self.config.history and previous is not None:
            history = (
                f"\nPrevious conclusion at the earlier timestep: {previous}.\n"
                "This is historical belief state, not an additional premise. Keep it only "
                "if the current premises still justify it.\n"
            )
        reviewer = (
            f"\nReviewer note to consider: {reviewer_note}\n"
            if reviewer_note else ""
        )
        prompt = f"""Choose the conclusion that necessarily follows in this Belief-R state.
Premises:
{_premises(scenario['observations'])}
Candidate conclusions:
{_options(scenario['hypotheses'])}
Evaluator assessment (it may be wrong, so reason critically):
{_evaluation_text(evaluation) if evaluation else 'No separate evaluation.'}
{history}{reviewer}
Use defeasible commonsense reasoning: a newly revealed prerequisite may require
retracting an earlier conclusion, while an independent alternative route should
not by itself invalidate that conclusion. Do not invent missing premises.
End with exactly one line: ANSWER: H1, ANSWER: H2, or ANSWER: H3"""
        votes = []
        for i in range(self.config.k):
            text = self._ask(
                prompt,
                seed=seed_base + round_no * 100 + i,
                stage=f"belief select {i + 1}/{self.config.k}",
                trace=trace,
            )
            votes.append({
                "answer": _tag(text, "ANSWER", {"H1", "H2", "H3"}),
                "text": text,
            })
        counts = Counter(vote["answer"] for vote in votes)
        maximum = max(counts.values())
        winners = [hid for hid, n in counts.items() if n == maximum]
        tie_breaker = None
        if len(winners) == 1:
            chosen = winners[0]
        else:
            tied = prompt + "\nVotes tied. Choose only among: " + ", ".join(winners)
            text = self._ask(
                tied,
                seed=seed_base + round_no * 100 + 99,
                stage="belief tie breaker",
                trace=trace,
            )
            chosen = _tag(text, "ANSWER", set(winners))
            tie_breaker = text
        return {
            "chosen": chosen,
            "counts": dict(counts),
            "votes": votes,
            "tie_breaker": tie_breaker,
        }

    def _reflect(self, scenario: dict, evaluation: dict, selection: dict,
                 round_no: int, trace: list) -> dict:
        previous = scenario["rounds"][-1]["final"] if scenario["rounds"] else None
        history = ""
        if self.config.history and previous is not None:
            history = (
                f"\nEarlier conclusion: {previous}. It is history, not evidence.\n"
            )
        prompt = f"""Critically review this Belief-R decision without any answer key.
Premises:
{_premises(scenario['observations'])}
Candidate conclusions:
{_options(scenario['hypotheses'])}
Evaluator assessment:
{_evaluation_text(evaluation) if evaluation else 'No separate evaluation.'}
Preliminary choice: {selection['chosen']}
Votes: {selection['counts']}{history}

Check for a concrete error: misuse of modus ponens/tollens, treating an
alternative route as a prerequisite, missing a commonsense additional condition,
or unjustified belief inertia/over-revision. Merely being uncertain is not enough.
End with exactly one line:
REVIEW: KEEP
or
REVIEW: REVISE"""
        text = self._ask(
            prompt,
            seed=3000 + round_no,
            stage="belief self-refine critique",
            trace=trace,
        )
        verdict = _tag(text, "REVIEW", {"KEEP", "REVISE"})
        note = "\n".join(
            line for line in text.splitlines()
            if not re.match(r"\s*REVIEW\s*:", line, re.I)
        ).strip()
        return {"verdict": verdict, "note": note, "text": text}

    def run_round(self, scenario: dict) -> dict:
        if scenario["rounds"] and scenario["rounds"][-1]["observations"] == scenario["observations"]:
            return scenario["rounds"][-1]
        if len(scenario["hypotheses"]) != 3:
            raise ValueError("Belief-R requires exactly three provided conclusions")

        round_no = len(scenario["rounds"]) + 1
        trace = []
        evaluation = (
            self._evaluate(scenario, round_no, trace)
            if self.config.evaluate else {}
        )
        selection = self._select(scenario, evaluation, round_no, trace)
        critique = (
            self._reflect(scenario, evaluation, selection, round_no, trace)
            if self.config.self_refine else
            {"verdict": "SKIP", "note": "", "text": ""}
        )
        revision = None
        final = selection["chosen"]
        if critique["verdict"] == "REVISE":
            new_eval = (
                self._evaluate(
                    scenario, round_no, trace,
                    correction=critique["note"], seed_base=4000,
                )
                if self.config.evaluate else {}
            )
            new_sel = self._select(
                scenario, new_eval, round_no, trace,
                reviewer_note=critique["note"], seed_base=5000,
            )
            revision = {"evaluation": new_eval, "selection": new_sel}
            final = new_sel["chosen"]

        previous = scenario["rounds"][-1]["final"] if scenario["rounds"] else None
        item = {
            "round": round_no,
            "observations": list(scenario["observations"]),
            "hypotheses": dict(scenario["hypotheses"]),
            "evaluation": evaluation,
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
