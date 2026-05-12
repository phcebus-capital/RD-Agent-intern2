"""
Hypothesis generation and Hypothesis→Experiment conversion for TX futures.

Reuses LLMHypothesisGen / LLMHypothesis2Experiment from the components layer;
only the context-preparation and JSON-parsing methods are futures-specific.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Tuple

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.futures.experiment import FuturesFactorExperiment
from rdagent.utils.agent.tpl import T


class FuturesFactorHypothesis(Hypothesis):
    """Hypothesis extended with a method_type label for exhaustion tracking."""

    def __init__(self, *args, method_type: str = "other", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.method_type: str = method_type


# ─────────────────────── HypothesisGen ──────────────────────────────────


def _compute_exhausted_methods(trace: Trace, max_tries: int) -> list[str]:
    """
    Return method_types that have been tried >= max_tries times with zero SOTA wins.

    A method_type is exhausted when it appears at least max_tries times in the
    trace history AND none of those trials had feedback.decision == True.
    """
    tries: Counter = Counter()
    wins: Counter = Counter()
    for exp, fb in trace.hist:
        mt = getattr(getattr(exp, "hypothesis", None), "method_type", None)
        if mt:
            tries[mt] += 1
            if fb and fb.decision:
                wins[mt] += 1
    return [mt for mt, count in tries.items() if count >= max_tries and wins[mt] == 0]


class FuturesFactorHypothesisGen(FactorHypothesisGen):
    """
    Proposes a new TX 1-min signal idea based on the experiment history (trace).
    """

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING

        hypothesis_and_feedback = (
            T("scenarios.futures.prompts:hypothesis_and_feedback").r(trace=trace)
            if trace.hist
            else "No previous experiments yet – this is the first round."
        )

        last_hypothesis_and_feedback = (
            T("scenarios.futures.prompts:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0],
                feedback=trace.hist[-1][1],
            )
            if trace.hist
            else "No previous experiments yet."
        )

        n_rounds = len(trace.hist)
        rag_hint = (
            "Start with simple, interpretable signals (momentum, VWAP deviation, volume breakout)."
            if n_rounds < 10
            else "The simple signals have been explored. Try more complex combinations or ML-based signals."
        )

        exhausted = _compute_exhausted_methods(trace, FUTURES_FACTOR_PROP_SETTING.max_method_tries)
        if exhausted:
            exhausted_note = (
                f"\n\nEXHAUSTED METHOD TYPES (do NOT propose these again — "
                f"tried {FUTURES_FACTOR_PROP_SETTING.max_method_tries}+ times without SOTA improvement): "
                + ", ".join(exhausted)
            )
            rag_hint += exhausted_note

        return {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "RAG": rag_hint,
            "hypothesis_output_format": T("scenarios.futures.prompts:hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.futures.prompts:hypothesis_specification").r(),
        }, True

    def convert_response(self, response: str) -> Hypothesis:
        d = json.loads(response)
        return FuturesFactorHypothesis(
            hypothesis=d.get("hypothesis", ""),
            reason=d.get("reason", ""),
            concise_reason=d.get("concise_reason", ""),
            concise_observation=d.get("concise_observation", ""),
            concise_justification=d.get("concise_justification", ""),
            concise_knowledge=d.get("concise_knowledge", ""),
            method_type=d.get("method_type", "other"),
        )


class FuturesFactorHypothesisGenFromStrategy(FuturesFactorHypothesisGen):
    """Seeds the first round with a user-provided strategy text, then iterates normally."""

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING
        self.initial_strategy: str = FUTURES_FACTOR_PROP_SETTING.initial_strategy

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        ctx, system = super().prepare_context(trace)
        if not self.initial_strategy:
            return ctx, system

        if not trace.hist:
            seed_text = (
                f"User-provided strategy direction:\n{self.initial_strategy}\n\n"
                "Use this as inspiration to propose a concrete, testable signal hypothesis. "
                "You may interpret it broadly — the goal is to explore the spirit of this "
                "direction, not to implement it literally."
            )
            ctx["hypothesis_and_feedback"] = seed_text
            ctx["last_hypothesis_and_feedback"] = seed_text
            ctx["RAG"] = (
                f"The user has suggested a strategy direction (below). "
                f"Let it guide your thinking, but feel free to adapt or generalise it.\n\n"
                f"--- User strategy ---\n{self.initial_strategy}\n--- End ---"
            )
        else:
            ctx["RAG"] = (
                f"The user's initial strategy direction (below) is a soft preference — "
                f"lean towards it when proposing new hypotheses, but do not feel constrained "
                f"if the experiment history points to a more promising path.\n\n"
                f"--- User strategy ---\n{self.initial_strategy}\n--- End ---\n\n"
                + ctx.get("RAG", "")
            )
        return ctx, system


# ──────────────────── Hypothesis2Experiment ──────────────────────────────

class FuturesFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    """
    Translates a Hypothesis into a FuturesFactorExperiment with FactorTask list.
    """

    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        if trace.hist:
            hypothesis_and_feedback = T(
                "scenarios.futures.prompts:hypothesis_and_feedback"
            ).r(trace=trace)
        else:
            hypothesis_and_feedback = "No previous experiments yet."

        return {
            "target_hypothesis": str(hypothesis),
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": T(
                "scenarios.futures.prompts:factor_experiment_output_format"
            ).r(),
            "target_list": [],
            "RAG": None,
        }, True

    def convert_response(
        self, response: str, hypothesis: Hypothesis, trace: Trace
    ) -> FuturesFactorExperiment:
        from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING
        from rdagent.scenarios.futures.experiment import FuturesFBWorkspace

        d = json.loads(response)
        tasks = []
        for signal_name, info in d.items():
            tasks.append(
                FactorTask(
                    factor_name=signal_name,
                    factor_description=info.get("description", ""),
                    factor_formulation=info.get("formulation", ""),
                    variables=info.get("variables", {}),
                )
            )

        exp = FuturesFactorExperiment(tasks, hypothesis=hypothesis)

        # Build the baseline: use custom factor.py if provided, else empty zero-signal baseline
        baseline_code = FUTURES_FACTOR_PROP_SETTING.baseline_factor
        if baseline_code:
            baseline_task = FactorTask(
                factor_name="custom_baseline",
                factor_description="User-provided baseline strategy",
                factor_formulation="See factor.py code",
                variables={},
            )
            baseline_ws = FuturesFBWorkspace(target_task=baseline_task)
            baseline_ws.inject_files(**{"factor.py": baseline_code})
            baseline_exp = FuturesFactorExperiment(
                sub_tasks=[baseline_task], hypothesis=None
            )
            baseline_exp.sub_workspace_list = [baseline_ws]
        else:
            baseline_exp = FuturesFactorExperiment(sub_tasks=[])

        # Chain of successful experiments (SOTA history) for the runner to reference
        exp.based_experiments = [baseline_exp] + [
            t[0]
            for t in trace.hist
            if t[1] and isinstance(t[0], FuturesFactorExperiment)
        ]

        # De-duplicate: skip tasks already explored in previous experiments
        seen_names = {
            sub_task.factor_name
            for based_exp in exp.based_experiments
            for sub_task in based_exp.sub_tasks
        }
        exp.sub_tasks = [t for t in tasks if t.factor_name not in seen_names]

        return exp
