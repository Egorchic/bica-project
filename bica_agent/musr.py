"""Task-specific MuSR reasoning built on the existing OllamaLLM interface.

No answer key is accepted by this module. It sees only public benchmark fields:
domain, context, question and choices.
"""

from __future__ import annotations

from collections import Counter
import re


DOMAIN_HINTS = {
    "murder_mysteries": (
        "A murderer should have means (access to the weapon), motive, and "
        "opportunity (access to the crime scene at the relevant time). "
        "Distinguish evidence that actually establishes these conditions from "
        "mere suspicion."
    ),
    "object_placements": (
        "Track where the relevant person believes the object is after every "
        "move. A person updates their belief only when they observe the move; "
        "otherwise they retain the last location they observed."
    ),
    "team_allocation": (
        "Infer each person's task skill and relevant teamwork quality. Compare "
        "the complete assignments, including the task that receives two people, "
        "and choose the assignment that makes the overall allocation strongest."
    ),
}


def _choices_text(choices: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"{i}. {choice}" for i, choice in enumerate(choices, 1))


def _answer(text: str, n_choices: int) -> int:
    found = []
    for line in text.splitlines():
        clean = line.strip().replace("**", "").strip("` ")
        m = re.fullmatch(r"ANSWER\s*:\s*(\d+)\.?", clean, re.I)
        if m:
            found.append(int(m.group(1)))
    if len(found) != 1 or not 1 <= found[0] <= n_choices:
        raise ValueError(
            f"Expected exactly one ANSWER: 1..{n_choices}, got: {text[:500]!r}"
        )
    return found[0] - 1


def _review(text: str) -> str:
    found = []
    for line in text.splitlines():
        clean = line.strip().replace("**", "").strip("` ")
        m = re.fullmatch(r"REVIEW\s*:\s*(KEEP|REVISE)\.?", clean, re.I)
        if m:
            found.append(m.group(1).upper())
    if len(found) != 1:
        raise ValueError(f"Expected exactly one REVIEW: KEEP/REVISE: {text[:500]!r}")
    return found[0]


def _task_text(domain: str, context: str, question: str, choices) -> str:
    if domain not in DOMAIN_HINTS:
        raise ValueError(f"Unknown MuSR domain: {domain}")
    return f"""Domain guidance:
{DOMAIN_HINTS[domain]}

Story:
{context}

Question:
{question}

Choices:
{_choices_text(choices)}"""


class MuSRReasoner:
    """M0 direct baseline and M1 structured evidence + voting + self-refine."""

    def __init__(self, llm):
        self.llm = llm

    def direct(
        self,
        *,
        domain: str,
        context: str,
        question: str,
        choices,
        seed_base: int,
    ) -> dict:
        task = _task_text(domain, context, question, choices)
        prompt = f"""Solve this MuSR multiple-choice reasoning problem.

{task}

Use only information supported by the story plus ordinary commonsense required
to interpret it. Choose exactly one option.
End your response with exactly one line:
ANSWER: <choice number>"""
        reply = self.llm.ask(prompt, seed=seed_base, stage="musr_direct")
        pred = _answer(reply.text, len(choices))
        return {
            "prediction": pred,
            "preliminary_prediction": pred,
            "evidence_analysis": None,
            "selection": {"votes": [pred], "vote_counts": {str(pred): 1}},
            "critique": {"verdict": "SKIP", "note": ""},
            "revision": None,
        }

    def structured(
        self,
        *,
        domain: str,
        context: str,
        question: str,
        choices,
        seed_base: int,
    ) -> dict:
        task = _task_text(domain, context, question, choices)

        evidence_prompt = f"""Analyze this MuSR problem before choosing an answer.

{task}

Build a compact evidence ledger. Extract only facts that can affect the answer,
track the multi-step dependencies needed for the question, and explicitly note
important uncertainties or conflicts. Do not choose an answer yet.
Keep the ledger concise: at most 12 bullet points."""
        evidence_reply = self.llm.ask(
            evidence_prompt, seed=seed_base + 100, stage="musr_evidence"
        )
        evidence = evidence_reply.text.strip()
        if not evidence:
            raise ValueError("Empty MuSR evidence analysis")

        selection_prompt = f"""Choose the best answer to this MuSR problem.

{task}

A previous analysis produced this evidence ledger:
{evidence}

The ledger may contain mistakes. Check it critically against the story before
deciding. Compare all choices and choose exactly one.
End your response with exactly one line:
ANSWER: <choice number>"""

        votes = []
        vote_texts = []
        for i in range(3):
            reply = self.llm.ask(
                selection_prompt,
                seed=seed_base + 200 + i,
                stage=f"musr_select_vote_{i+1}",
            )
            votes.append(_answer(reply.text, len(choices)))
            vote_texts.append(reply.text)

        counts = Counter(votes)
        best = max(counts.values())
        winners = [idx for idx, count in counts.items() if count == best]
        tie_breaker = None
        if len(winners) == 1:
            preliminary = winners[0]
        else:
            allowed = ", ".join(str(x + 1) for x in sorted(winners))
            tie_prompt = (
                selection_prompt
                + f"\nThe three votes tied. Choose only among: {allowed}.\n"
                + "End with exactly one line: ANSWER: <choice number>"
            )
            reply = self.llm.ask(
                tie_prompt,
                seed=seed_base + 299,
                stage="musr_tie_breaker",
            )
            preliminary = _answer(reply.text, len(choices))
            if preliminary not in winners:
                raise ValueError("Tie breaker selected an option outside the tied set")
            tie_breaker = reply.text

        critique_prompt = f"""Critically review a preliminary MuSR decision.

{task}

Evidence ledger:
{evidence}

Vote counts by zero-based internal option index:
{dict(counts)}
Preliminary choice: {preliminary + 1}

Check for a concrete reasoning error: a missed story fact, incorrect state
tracking, unsupported inference, inconsistent assignment, or a failure to
compare a decisive alternative. Do not revise merely because the task is hard.
Write a concise critique, then end with exactly one line:
REVIEW: KEEP
or
REVIEW: REVISE"""
        critique_reply = self.llm.ask(
            critique_prompt,
            seed=seed_base + 300,
            stage="musr_self_refine_critique",
        )
        verdict = _review(critique_reply.text)
        critique_note = "\n".join(
            line
            for line in critique_reply.text.splitlines()
            if not re.match(r"\s*REVIEW\s*:", line, re.I)
        ).strip()

        final = preliminary
        revision = None
        if verdict == "REVISE":
            revision_prompt = f"""Re-solve this MuSR problem after reviewer feedback.

{task}

Earlier evidence ledger:
{evidence}

Earlier choice: {preliminary + 1}
Reviewer feedback:
{critique_note}

Correct the concrete error identified by the reviewer. Re-check the decisive
facts against the story and choose exactly one option.
End your response with exactly one line:
ANSWER: <choice number>"""
            revision_reply = self.llm.ask(
                revision_prompt,
                seed=seed_base + 400,
                stage="musr_revision_select",
            )
            final = _answer(revision_reply.text, len(choices))
            revision = {
                "prediction": final,
                "response": revision_reply.text,
            }

        return {
            "prediction": final,
            "preliminary_prediction": preliminary,
            "evidence_analysis": evidence,
            "selection": {
                "votes": votes,
                "vote_counts": {str(k): v for k, v in counts.items()},
                "vote_responses": vote_texts,
                "tie_breaker": tie_breaker,
            },
            "critique": {
                "verdict": verdict,
                "note": critique_note,
                "response": critique_reply.text,
            },
            "revision": revision,
        }
