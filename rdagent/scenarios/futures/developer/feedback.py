"""
Feedback generator for TX futures signal experiments.

Compares current experiment's backtest metrics against SOTA,
asks LLM to evaluate the hypothesis and propose next direction.
"""

from __future__ import annotations

import json
from typing import Dict

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.futures.developer.factor_runner import format_result
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T


class FuturesFactorExperiment2Feedback(Experiment2Feedback):
    """
    Generates structured LLM feedback after a futures strategy experiment.

    Compares current Sharpe/drawdown vs SOTA and decides whether this
    experiment should replace the best-known result.
    """

    def generate_feedback(
        self, exp: Experiment, trace: Trace
    ) -> HypothesisFeedback:
        hypothesis = exp.hypothesis
        logger.info("Generating feedback for futures experiment …")

        current_result = exp.result
        sota_result = (
            exp.based_experiments[-1].result
            if exp.based_experiments
            else None
        )

        current_str = format_result(current_result)
        sota_str = format_result(sota_result)
        combined_result = (
            f"Current experiment: {current_str}\n"
            f"SOTA (best so far):  {sota_str}"
        )

        tasks_info = [
            task.get_task_information_and_implementation_result()
            for task in exp.sub_tasks
        ]

        sys_prompt = T("scenarios.futures.prompts:factor_feedback_generation.system").r(
            scenario=self.scen.get_scenario_all_desc()
        )
        usr_prompt = T("scenarios.futures.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis.hypothesis,
            task_details=tasks_info,
            combined_result=combined_result,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool],
        )
        resp = json.loads(response)

        return HypothesisFeedback(
            observations=resp.get("Observations", "No observations provided."),
            hypothesis_evaluation=resp.get("Feedback for Hypothesis", "No feedback provided."),
            new_hypothesis=resp.get("New Hypothesis", ""),
            reason=resp.get("Reasoning", ""),
            decision=convert2bool(resp.get("Replace Best Result", "false")),
        )
