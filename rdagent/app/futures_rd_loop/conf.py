"""
Configuration for the TX futures RD-Agent loop.

All settings can be overridden via FUTURES_FACTOR_ environment variables.

Example .env file:
  FUTURES_FACTOR_train_start=2020-01-30
  FUTURES_FACTOR_train_end=2022-12-31
  FUTURES_FACTOR_valid_start=2023-01-01
  FUTURES_FACTOR_valid_end=2024-06-30
  FUTURES_FACTOR_test_start=2024-07-01

  FUTURES_CoSTEER_data_folder=git_ignore_folder/futures_source_data
  FUTURES_CoSTEER_data_folder_debug=git_ignore_folder/futures_source_data_debug
"""

from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class FuturesFactorBasePropSetting(BasePropSetting):
    model_config = SettingsConfigDict(
        env_prefix="FUTURES_FACTOR_", protected_namespaces=()
    )

    # ── Scenario ──────────────────────────────────────────────────────
    scen: str = "rdagent.scenarios.futures.experiment.FuturesFactorScenario"

    # ── Proposal ──────────────────────────────────────────────────────
    hypothesis_gen: str = (
        "rdagent.scenarios.futures.proposal.FuturesFactorHypothesisGen"
    )
    hypothesis2experiment: str = (
        "rdagent.scenarios.futures.proposal.FuturesFactorHypothesis2Experiment"
    )

    # ── Developer ─────────────────────────────────────────────────────
    coder: str = (
        "rdagent.scenarios.futures.developer.factor_coder.FuturesFactorCoSTEER"
    )
    runner: str = (
        "rdagent.scenarios.futures.developer.factor_runner.FuturesFactorRunner"
    )
    summarizer: str = (
        "rdagent.scenarios.futures.developer.feedback.FuturesFactorExperiment2Feedback"
    )

    # ── Evolution ─────────────────────────────────────────────────────
    evolving_n: int = 10

    # ── Data splits ───────────────────────────────────────────────────
    train_start: str = "2020-01-02"
    train_end: str = "2022-12-31"
    valid_start: str = "2023-01-01"
    valid_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    test_end: Optional[str] = "2026-04-15"


FUTURES_FACTOR_PROP_SETTING = FuturesFactorBasePropSetting()
