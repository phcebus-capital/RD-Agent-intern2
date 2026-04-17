"""
Runner for TX futures signal experiments.

Replaces Qlib's docker-based backtest with a pure-pandas vectorized backtest:
  1. Execute each sub_workspace to get a 1-min signal series (full data)
  2. Combine signals from all sub_tasks (average)
  3. Run vectorized backtest on the TEST period
  4. Store metrics dict in exp.result

Backtest model:
  - Product      : FITX*1.TF (台股指數近月, TX)
  - Bar size     : 1-minute
  - Position     : sign(signal) → +1 Long / -1 Short / 0 Flat
  - Execution    : at open of next 1-min bar (市價+1檔 for buys, 市價-1檔 for sells)
  - Cost per side: 1.525 pts (0.525 pts commission = 105 NTD/lot ÷ 200 NTD/pt + 1 pt slippage)
  - Daily reset  : 每日部位歸零 = No (positions carry across sessions and days)
  - Date range   : 2020-01-02 ~ 2026-04-15
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.futures.developer.factor_coder import FUTURES_COSTEER_SETTINGS
from rdagent.scenarios.futures.experiment import FuturesFactorExperiment


# ─────────────────────── Backtest engine ─────────────────────────────────

def run_backtest(df: pd.DataFrame, signal: pd.Series, cost_pts: float = 1.525) -> dict:
    """
    Vectorized position-based backtest for 1-min TX futures.

    Parameters
    ----------
    df : pd.DataFrame
        1-min OHLCV data with 'session' column and DatetimeIndex.
    signal : pd.Series
        Raw signal values (same index as df). Positive=long, negative=short.
    cost_pts : float
        Cost per side in index points:
          0.525 pts = 105 NTD commission ÷ 200 NTD/pt
          1.000 pt  = 市價±1檔 slippage (預設買進市價+1檔 / 賣出市價-1檔)
          Total     = 1.525 pts/side  (default)

    Returns
    -------
    dict with keys: sharpe, annual_return_pts, max_drawdown_pts,
                    n_trades, day_sharpe, night_sharpe
    """
    # ── Align ──────────────────────────────────────────────────────────
    sig = signal.reindex(df.index).fillna(0)
    pos = np.sign(sig)

    # 每日部位歸零 = No: positions carry freely across session and day boundaries.
    pos_adj = pos.copy()

    # ── Price change and position change ───────────────────────────────
    price_chg = df["close"].diff().fillna(0.0)
    pos_chg = pos_adj.diff().fillna(pos_adj)

    # Holding P&L: previous position × current price change
    hold_pnl = pos_adj.shift(1).fillna(0.0) * price_chg

    # Transaction cost: paid whenever position changes
    cost = pos_chg.abs() * cost_pts

    net_pnl = hold_pnl - cost

    # ── Cumulative metrics ─────────────────────────────────────────────
    cum_pnl = net_pnl.cumsum()
    total_pts = float(cum_pnl.iloc[-1]) if len(cum_pnl) > 0 else 0.0

    n_days = max(1, (df.index[-1] - df.index[0]).days)
    n_years = n_days / 365.0
    annual_return = total_pts / n_years

    # Max drawdown
    max_drawdown = float((cum_pnl - cum_pnl.cummax()).min())

    # Annualised Sharpe (assume ~300 active 1-min bars per trading day × 250 days)
    bars_per_year = 300 * 250
    std = float(net_pnl.std())
    sharpe = float(net_pnl.mean() / std * np.sqrt(bars_per_year)) if std > 0 else 0.0

    # Trade count: half the number of position changes
    n_trades = int((pos_chg.abs() > 0).sum() // 2)

    # ── Session breakdown ──────────────────────────────────────────────
    def _session_sharpe(mask: pd.Series, est_bars_per_year: int) -> float:
        s = net_pnl[mask]
        sd = float(s.std())
        return float(s.mean() / sd * np.sqrt(est_bars_per_year)) if sd > 0 else 0.0

    day_mask = df["session"] == "day"
    night_mask = df["session"] == "night"
    day_sharpe = _session_sharpe(day_mask, 60 * 250)     # ~60 min/day session × 250 days
    night_sharpe = _session_sharpe(night_mask, 240 * 250) # ~240 min/night × 250 days

    return {
        "sharpe": round(sharpe, 3),
        "annual_return_pts": round(annual_return, 1),
        "max_drawdown_pts": round(max_drawdown, 1),
        "n_trades": n_trades,
        "day_sharpe": round(day_sharpe, 3),
        "night_sharpe": round(night_sharpe, 3),
    }


def format_result(result: dict | None) -> str:
    if result is None:
        return "No result (execution failed)."
    return (
        f"Sharpe={result['sharpe']:.3f} | "
        f"Annual={result['annual_return_pts']:.0f}pts | "
        f"MaxDD={result['max_drawdown_pts']:.0f}pts | "
        f"Trades={result['n_trades']} | "
        f"DaySharpe={result['day_sharpe']:.3f} | "
        f"NightSharpe={result['night_sharpe']:.3f}"
    )


# ─────────────────────── Runner ──────────────────────────────────────────

class FuturesFactorRunner(CachedRunner[FuturesFactorExperiment]):
    """
    Executes factor code on the full dataset, runs pandas backtest,
    and stores metrics in exp.result.

    No Qlib dependency.
    """

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: FuturesFactorExperiment) -> FuturesFactorExperiment:
        from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING

        conf = FUTURES_FACTOR_PROP_SETTING

        # ── Recursively ensure baseline is evaluated ──────────────────
        if (
            exp.based_experiments
            and exp.based_experiments[-1].result is None
            and exp.based_experiments[-1].sub_workspace_list  # skip empty baseline
        ):
            logger.info("Evaluating baseline experiment first …")
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        # ── Execute each sub_workspace on full data ───────────────────
        signals: list[pd.Series] = []
        for i, ws in enumerate(exp.sub_workspace_list):
            if ws is None:
                logger.warning(f"sub_workspace_list[{i}] is None — skipping.")
                continue
            logger.info(f"Executing workspace {i + 1}/{len(exp.sub_workspace_list)} (Full data) …")
            feedback, signal = ws.execute(data_type="Full")
            exp.stdout += f"\n--- Workspace {i} ---\n{feedback}"
            if signal is None:
                logger.warning(f"Workspace {i} returned no signal.")
                continue
            signals.append(signal)

        if not signals:
            raise FactorEmptyError(
                "All sub_workspaces failed to produce a valid signal. "
                f"Stdout:\n{exp.stdout}"
            )

        # ── Combine signals (mean across tasks) ───────────────────────
        combined_signal = pd.concat(signals, axis=1).mean(axis=1)
        combined_signal.name = "signal"

        # ── Load full 1-min data ──────────────────────────────────────
        data_path = Path(FUTURES_COSTEER_SETTINGS.data_folder) / "data.parquet"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Full dataset not found at {data_path}. "
                "Run rdagent/scenarios/futures/prepare_data.py first."
            )
        df_full = pd.read_parquet(data_path)

        # ── Restrict to test period ───────────────────────────────────
        test_start = conf.test_start
        test_end = conf.test_end  # may be None

        df_test = df_full.loc[test_start:test_end] if test_end else df_full.loc[test_start:]
        sig_test = combined_signal.reindex(df_test.index).fillna(0)

        if df_test.empty:
            raise FactorEmptyError(f"No data in test period (>= {test_start}).")

        # ── Run backtest ──────────────────────────────────────────────
        logger.info(
            f"Running backtest on test period "
            f"{df_test.index[0].date()} → {df_test.index[-1].date()} …"
        )
        result = run_backtest(df_test, sig_test)
        result["test_period"] = (
            f"{df_test.index[0].date()} ~ {df_test.index[-1].date()}"
        )

        logger.info(f"Backtest result: {format_result(result)}")
        exp.result = result
        return exp
