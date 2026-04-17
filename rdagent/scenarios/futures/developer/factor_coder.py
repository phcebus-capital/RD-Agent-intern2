"""
CoSTEER-based coder for TX futures signals.

Key differences from the stock factor coder:
1. Uses FUTURES_CoSTEER_ env-prefix settings → points to futures data folders
2. FuturesMultiProcessEvolvingStrategy creates FuturesFBWorkspace objects
   instead of FactorFBWorkspace, so that execute() uses futures data paths
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.factor_coder.config import CoSTEERSettings, FactorCoSTEERSettings
from rdagent.components.coder.factor_coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy,
)
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.scenarios.futures.developer.evaluators import FuturesFactorEvaluatorForCoder
from rdagent.scenarios.futures.experiment import FuturesFBWorkspace


# ──────────────────── Settings ───────────────────────────────────────────

class FuturesCoSTEERSettings(FactorCoSTEERSettings):
    """
    Same as FactorCoSTEERSettings but reads from FUTURES_CoSTEER_ env prefix.

    Set before running:
      export FUTURES_CoSTEER_data_folder=/path/to/git_ignore_folder/futures_source_data
      export FUTURES_CoSTEER_data_folder_debug=/path/to/git_ignore_folder/futures_source_data_debug
    """

    model_config = SettingsConfigDict(env_prefix="FUTURES_CoSTEER_", protected_namespaces=())

    data_folder: str = "git_ignore_folder/futures_source_data"
    data_folder_debug: str = "git_ignore_folder/futures_source_data_debug"
    file_based_execution_timeout: int = 600
    python_bin: str = "python"


FUTURES_COSTEER_SETTINGS = FuturesCoSTEERSettings()


# ──────────────────── Evolving Strategy ──────────────────────────────────

class FuturesMultiProcessEvolvingStrategy(FactorMultiProcessEvolvingStrategy):
    """
    Same code-generation logic as the stock factor strategy, but creates
    FuturesFBWorkspace (futures data path) instead of FactorFBWorkspace.
    """

    def assign_code_list_to_evo(self, code_list, evo) -> None:
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FuturesFBWorkspace(
                    target_task=evo.sub_tasks[index]
                )
            files = (
                code_list[index]
                if isinstance(code_list[index], dict)
                else {"factor.py": code_list[index]}
            )
            # Remove change-summary key if present (not a file)
            files.pop(self.KEY_CHANGE_SUMMARY, None)
            evo.sub_workspace_list[index].inject_files(**files)
        return evo


# ──────────────────── CoSTEER Coder ──────────────────────────────────────

class FuturesFactorCoSTEER(CoSTEER):
    """
    CoSTEER-based coder that generates and evolves TX 1-min signal code.

    Uses:
    - FuturesCoSTEERSettings   → data paths point to futures parquet files
    - FuturesMultiProcessEvolvingStrategy → creates FuturesFBWorkspace
    - FactorEvaluatorForCoder  → validates code runs and produces valid output
    """

    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        setting = FUTURES_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(FuturesFactorEvaluatorForCoder(scen=scen), scen=scen)
        es = FuturesMultiProcessEvolvingStrategy(scen=scen, settings=setting)
        super().__init__(
            *args,
            settings=setting,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            **kwargs,
        )

    def develop(self, exp: Experiment) -> Experiment:
        try:
            exp = super().develop(exp)
        finally:
            if hasattr(self, "evolve_agent") and self.evolve_agent.evolving_trace:
                es = self.evolve_agent.evolving_trace[-1]
                exp.prop_dev_feedback = es.feedback
        return exp
