"""
Custom evaluators for TX 1-min futures signals.

Problem with the stock Qlib evaluators (FactorEvaluatorForCoder):
  1. FactorDatetimeDailyEvaluator rejects 1-minute data as "definitely wrong",
     because it expects daily frequency for stock factors.
  2. FactorValueEvaluator._get_df() converts the signal pd.Series to a DataFrame
     with column name 'source_factor', then FactorOutputFormatEvaluator sees a
     DataFrame and flags it — this is the evaluator's own conversion artifact,
     not a real bug in the user's code.

Fix: FuturesFactorValueEvaluator checks the raw signal directly (no DataFrame
conversion), and explicitly validates 1-minute frequency as correct.
"""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedbackDeprecated,
)
from rdagent.components.coder.factor_coder.eva_utils import (
    FactorCodeEvaluator,
    FactorEvaluator,
    FactorFinalDecisionEvaluator,
)
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Workspace

FactorSingleFeedback = CoSTEERSingleFeedbackDeprecated


class FuturesFactorValueEvaluator(FactorEvaluator):
    """
    Value evaluator for TX 1-min futures signals.

    Receives the already-executed signal (pd.Series | None) directly, so the
    subprocess is not run a second time.

    Checks:
      1. Output is pd.Series (not a DataFrame).
      2. No infinite values.
      3. Index is 1-minute DatetimeIndex (positive check — NOT a daily check).

    Returns (feedback_str, decision):
      decision = False  →  hard failure (wrong type, inf, or wrong frequency)
      decision = None   →  no hard failure; delegate to final_decision_evaluator
    """

    def evaluate_signal(
        self, signal: pd.Series | None
    ) -> Tuple[str, bool | None]:
        if signal is None:
            return (
                "No signal produced. Verify result.h5 is written with key='signal'.",
                False,
            )

        conclusions: list[str] = []
        decision: bool | None = None

        # 1. Type: must be pd.Series
        if not isinstance(signal, pd.Series):
            conclusions.append(
                f"Output is {type(signal).__name__}, expected pd.Series. "
                "Call .squeeze() or select a single column before the save block."
            )
            decision = False
        else:
            conclusions.append("Output is pd.Series — correct type.")

        # 2. No infinite values
        if isinstance(signal, pd.Series):
            inf_count = int(signal.isin([float("inf"), -float("inf")]).sum())
            if inf_count > 0:
                conclusions.append(
                    f"Signal has {inf_count} infinite value(s). Replace with finite values."
                )
                decision = False
            else:
                conclusions.append("No infinite values — OK.")

        # 3. Frequency: must be 1-minute (NOT daily)
        if (
            isinstance(signal, pd.Series)
            and isinstance(signal.index, pd.DatetimeIndex)
            and len(signal) > 1
        ):
            diffs = pd.Series(signal.index).diff().dropna().unique()
            if pd.Timedelta(minutes=1) in diffs:
                conclusions.append("Signal is at 1-minute frequency — correct.")
            else:
                conclusions.append(
                    "Signal does not appear to be at 1-minute frequency. "
                    "Do NOT resample or aggregate to daily; the signal must cover "
                    "every 1-min bar in df (same index as the input data)."
                )
                decision = False

        return "\n".join(conclusions), decision

    # Keep the standard evaluate() signature so this class is drop-in compatible,
    # but the actual logic lives in evaluate_signal().
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace = None,
        version: int = 1,
        **kwargs,
    ) -> Tuple[str, bool | None]:
        _, signal = implementation.execute()
        return self.evaluate_signal(signal)


class FuturesFactorEvaluatorForCoder(CoSTEEREvaluator):
    """
    Drop-in replacement for FactorEvaluatorForCoder tailored to futures.

    Key differences:
    - Uses FuturesFactorValueEvaluator (skips FactorDatetimeDailyEvaluator and
      avoids the Series→DataFrame conversion artifact).
    - Calls implementation.execute() exactly once; passes the signal directly
      to the value evaluator so the subprocess is not invoked twice.
    - gt_implementation is always None for futures (no ground truth).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.value_evaluator = FuturesFactorValueEvaluator(self.scen)
        self.code_evaluator = FactorCodeEvaluator(self.scen)
        self.final_decision_evaluator = FactorFinalDecisionEvaluator(self.scen)

    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        gt_implementation: Workspace = None,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorSingleFeedback:
        if implementation is None:
            return None

        task_info = target_task.get_task_information()

        # Fast-path: already solved / permanently failed
        if queried_knowledge is not None:
            if task_info in queried_knowledge.success_task_to_knowledge_dict:
                return queried_knowledge.success_task_to_knowledge_dict[task_info].feedback
            if task_info in queried_knowledge.failed_task_info_set:
                return FactorSingleFeedback(
                    execution_feedback="This task has failed too many times, skip implementation.",
                    value_generated_flag=False,
                    code_feedback="This task has failed too many times, skip code evaluation.",
                    value_feedback="This task has failed too many times, skip value evaluation.",
                    final_decision=False,
                    final_feedback="This task has failed too many times, skip final decision evaluation.",
                    final_decision_based_on_gt=False,
                )

        factor_feedback = FactorSingleFeedback()
        factor_feedback.final_decision_based_on_gt = False  # no GT for futures

        # ── 1. Execute once ──────────────────────────────────────────────
        execution_feedback, gen_signal = implementation.execute()
        execution_feedback = re.sub(
            r"(?<=\D)(,\s+-?\d+\.\d+){50,}(?=\D)", ", ", execution_feedback
        )
        factor_feedback.execution_feedback = "\n".join(
            line for line in execution_feedback.split("\n")
            if "warning" not in line.lower()
        )

        # ── 2. Value evaluation (futures-specific, no double execute) ────
        if gen_signal is None:
            factor_feedback.value_feedback = (
                "No signal produced. Cannot evaluate output."
            )
            factor_feedback.value_generated_flag = False
            decision_from_value_check = None
        else:
            factor_feedback.value_generated_flag = True
            factor_feedback.value_feedback, decision_from_value_check = (
                self.value_evaluator.evaluate_signal(gen_signal)
            )

        # ── 3. Code feedback + final decision ────────────────────────────
        if decision_from_value_check is True:
            factor_feedback.code_feedback = "Value checks passed — no code critics."
            factor_feedback.final_decision = True
            factor_feedback.final_feedback = "Value evaluation passed."

        elif decision_from_value_check is False:
            factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                target_task=target_task,
                implementation=implementation,
                execution_feedback=factor_feedback.execution_feedback,
                value_feedback=factor_feedback.value_feedback,
                gt_implementation=None,
            )
            factor_feedback.final_decision = False
            factor_feedback.final_feedback = "Value evaluation failed."

        else:
            # decision_from_value_check is None → ask LLM for final decision
            factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                target_task=target_task,
                implementation=implementation,
                execution_feedback=factor_feedback.execution_feedback,
                value_feedback=factor_feedback.value_feedback,
                gt_implementation=None,
            )
            factor_feedback.final_decision, factor_feedback.final_feedback = (
                self.final_decision_evaluator.evaluate(
                    target_task=target_task,
                    execution_feedback=factor_feedback.execution_feedback,
                    value_feedback=factor_feedback.value_feedback,
                    code_feedback=factor_feedback.code_feedback,
                )
            )

        return factor_feedback
