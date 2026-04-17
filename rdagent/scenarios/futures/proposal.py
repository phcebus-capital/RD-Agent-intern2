"""
Hypothesis generation and Hypothesis→Experiment conversion for TX futures.

Reuses LLMHypothesisGen / LLMHypothesis2Experiment from the components layer;
only the context-preparation and JSON-parsing methods are futures-specific.
"""

from __future__ import annotations

import json
from typing import Tuple

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.futures.experiment import FuturesFactorExperiment
from rdagent.utils.agent.tpl import T

FuturesFactorHypothesis = Hypothesis


# ─────────────────────── HypothesisGen ──────────────────────────────────

class FuturesFactorHypothesisGen(FactorHypothesisGen):
    """
    Proposes a new TX 1-min signal idea based on the experiment history (trace).
    """

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
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
        )


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

        # Chain of successful experiments (SOTA history) for the runner to reference
        exp.based_experiments = [FuturesFactorExperiment(sub_tasks=[])] + [
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
