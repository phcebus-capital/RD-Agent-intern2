"""
Taiwan Futures (TX) 1-min strategy scenario.

Data contract for LLM-generated factor.py:
  - Input : data.parquet  → pd.DataFrame, index=DatetimeIndex (1-min),
            columns: open, high, low, close, volume, session
  - Output: result.h5     → pd.Series (key="signal"), same index as input,
            positive = long signal, negative = short, ~0 = flat
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import pandas as pd
from filelock import FileLock

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.experiment import Experiment, FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils.agent.tpl import T


# ─────────────────────────── Workspace ──────────────────────────────────

class FuturesFBWorkspace(FBWorkspace):
    """
    Workspace for a single futures signal (factor.py).

    execute(data_type):
      - "Debug" → links data from FUTURES_COSTEER_SETTINGS.data_folder_debug
      - "Full"  → links data from FUTURES_COSTEER_SETTINGS.data_folder
      Returns (feedback_str, pd.Series | None)
    """

    EXEC_SUCCESS = "Execution succeeded without error."
    CODE_NOT_SET = "Code not set."
    OUTPUT_NOT_FOUND = "\nExpected output file (result.h5) not found."
    OUTPUT_FOUND = "\nExpected output file (result.h5) found."

    def __init__(self, *args, raise_exception: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    def hash_func(self, data_type: str = "Debug") -> str:
        code = self.file_dict.get("factor.py", "")
        return md5_hash(data_type + code) if code else None

    def execute(self, data_type: str = "Debug") -> Tuple[str, pd.Series | None]:
        # Import here to avoid circular imports at module load time
        from rdagent.scenarios.futures.developer.factor_coder import FUTURES_COSTEER_SETTINGS

        if "factor.py" not in (self.file_dict or {}):
            return self.CODE_NOT_SET, None

        settings = FUTURES_COSTEER_SETTINGS
        data_folder = Path(
            settings.data_folder_debug if data_type == "Debug" else settings.data_folder
        )

        self.workspace_path.mkdir(parents=True, exist_ok=True)
        code_path = self.workspace_path / "factor.py"
        code_path.write_text(self._fix_code(self.file_dict["factor.py"]))

        # Symlink every file in the data folder into the workspace.
        # Always replace stale symlinks so that switching between data_type="Debug"
        # and data_type="Full" actually changes which parquet file is visible to
        # factor.py.  Without this, a debug-mode symlink left by the evaluator
        # would satisfy `dest.exists()` and silently prevent the full-data symlink
        # from being created, causing factor.py to run on debug data in "Full" mode
        # and producing a signal that covers only the debug period (~60 days).
        # When that short signal is reindexed to the test period it becomes all-zeros,
        # so the backtest reports Sharpe=0 and zero trades.
        for src in data_folder.glob("*"):
            dest = self.workspace_path / src.name
            if dest.is_symlink():
                dest.unlink()  # Remove stale symlink from a previous data_type
            if not dest.exists():
                try:
                    os.symlink(src.resolve(), dest)
                except OSError:
                    import shutil
                    shutil.copy2(src, dest)

        feedback = self.EXEC_SUCCESS
        exec_ok = False

        with FileLock(str(self.workspace_path / "execution.lock")):
            try:
                subprocess.check_output(
                    f"{settings.python_bin} factor.py",
                    shell=True,
                    cwd=self.workspace_path,
                    stderr=subprocess.STDOUT,
                    timeout=settings.file_based_execution_timeout,
                )
                exec_ok = True
            except subprocess.CalledProcessError as e:
                output = e.output.decode(errors="replace")
                feedback = output[:2000] if len(output) > 2000 else output
            except subprocess.TimeoutExpired:
                feedback = f"Timeout after {settings.file_based_execution_timeout}s."

            result_path = self.workspace_path / "result.h5"
            if result_path.exists() and exec_ok:
                try:
                    signal = pd.read_hdf(result_path, key="signal")
                    feedback += self.OUTPUT_FOUND
                    return feedback, signal
                except Exception as e:
                    feedback += f"\nError reading result.h5: {e}"
                    return feedback, None
            else:
                feedback += self.OUTPUT_NOT_FOUND
                return feedback, None

    @staticmethod
    def _fix_code(code: str) -> str:
        """Auto-fix common LLM typos that cause SyntaxError."""
        # Fix 'asert' → 'assert' (most common LLM typo)
        fixed = re.sub(r'\basert\b', 'assert', code)
        try:
            ast.parse(fixed)
            return fixed
        except SyntaxError:
            return code  # Return original; let subprocess capture the real error

    @property
    def all_codes(self) -> str:
        return self.file_dict.get("factor.py", "")

    def __str__(self) -> str:
        name = getattr(getattr(self, "target_task", None), "factor_name", "?")
        return f"FuturesFBWorkspace[{name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()


# ─────────────────────────── Experiment ─────────────────────────────────

class FuturesFactorExperiment(Experiment[FactorTask, FuturesFBWorkspace, FuturesFBWorkspace]):
    """
    An RD-Agent experiment for Taiwan Futures 1-min signal research.

    result (set by FuturesFactorRunner):
        {
          "sharpe": float,
          "annual_return_pts": float,
          "max_drawdown_pts": float,
          "n_trades": int,
          "day_sharpe": float,
          "night_sharpe": float,
          "test_period": str,
        }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stdout: str = ""


# ─────────────────────────── Scenario ───────────────────────────────────

class FuturesFactorScenario(Scenario):
    """Scenario descriptor for TX 1-min futures signal research."""

    def __init__(self) -> None:
        super().__init__()
        self._background = deepcopy(
            T("scenarios.futures.prompts:futures_factor_background").r()
        )
        self._source_data = deepcopy(
            T("scenarios.futures.prompts:futures_factor_source_data").r()
        )
        self._output_format = deepcopy(
            T("scenarios.futures.prompts:futures_factor_output_format").r()
        )
        self._interface = deepcopy(
            T("scenarios.futures.prompts:futures_factor_interface").r()
        )
        self._simulator = deepcopy(
            T("scenarios.futures.prompts:futures_factor_simulator").r()
        )

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return self._source_data

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return "Taiwan Futures (TX) 1-min K Bar Strategy Research"

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        if simple_background:
            return f"Background of the scenario:\n{self.background}"
        return (
            f"Background of the scenario:\n{self.background}\n\n"
            f"The source data you can use:\n{self.get_source_data_desc(task)}\n\n"
            f"The interface you should follow to write the runnable code:\n{self.interface}\n\n"
            f"The output format:\n{self.output_format}\n\n"
            f"The backtester that evaluates your signal:\n{self.simulator}\n"
        )

    def get_runtime_environment(self) -> str:
        return (
            "Python 3.10+, pandas, numpy, scipy. "
            "No external market data APIs are available inside the execution sandbox."
        )
