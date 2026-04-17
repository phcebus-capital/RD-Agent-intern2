"""
RD-Agent loop for TX 1-min futures strategy research.

Usage:
  # First, prepare data once:
  python rdagent/scenarios/futures/prepare_data.py

  # Then run the loop:
  python rdagent/app/futures_rd_loop/factor.py

  # Or resume a session:
  python rdagent/app/futures_rd_loop/factor.py --path LOG_PATH/__session__/1/0_propose

Required env vars (set in .env or shell):
  FUTURES_CoSTEER_data_folder=git_ignore_folder/futures_source_data
  FUTURES_CoSTEER_data_folder_debug=git_ignore_folder/futures_source_data_debug
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

import fire

from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import CoderError, FactorEmptyError
from rdagent.log import rdagent_logger as logger


class FuturesFactorRDLoop(RDLoop):
    """RD-Agent loop for TX 1-min futures strategy research."""

    skip_loop_error = (FactorEmptyError, CoderError)
    skip_loop_error_stepname = "feedback"

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            raise FactorEmptyError("Runner returned None — backtest failed.")
        logger.log_object(exp, tag="runner result")
        return exp


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: bool = True,
    checkout_path: Optional[str] = None,
) -> None:
    """
    Auto R&D evolving loop for TX 1-min futures strategy research.

    Args:
        path         : Resume from an existing session directory.
        step_n       : Run exactly this many steps, then stop.
        loop_n       : Run exactly this many full loops, then stop.
        all_duration : Stop after this wall-clock duration (e.g. "2h", "30m").
        checkout     : Whether to checkout the best result at end.
        checkout_path: Override path for best-result checkout.
    """
    if checkout_path is not None:
        checkout = Path(checkout_path)

    if path is None:
        loop = FuturesFactorRDLoop(FUTURES_FACTOR_PROP_SETTING)
    else:
        loop = FuturesFactorRDLoop.load(path, checkout=checkout)

    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
