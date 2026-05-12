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
  - Execution    : next bar open (signal confirmed at close of bar t → trade at open of bar t+1)
  - Cost per side: 1.525 pts (commission 105 NTD + slippage 200 NTD = 305 NTD ÷ 200 NTD/pt)
  - Daily reset  : 每日部位歸零 = No (positions carry across sessions and days)
  - Date range   : 2020-01-01 ~ 2026-04-15
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.futures.developer.factor_coder import FUTURES_COSTEER_SETTINGS
from rdagent.scenarios.futures.experiment import FuturesFactorExperiment


# ─────────────────────── Look-ahead bias detection ───────────────────────

def _detect_lookahead_bias(code: str) -> bool:
    """
    Detect common look-ahead bias patterns in factor code.

    Flags: groupby(date).transform() on OHLCV columns, which uses the
    full day's high/low/open/close at every bar — information not yet
    available at the time the bar is generated.
    """
    # Pattern: .groupby(<anything containing date/Date>).transform(
    patterns = [
        r'\.groupby\s*\([^)]*\.date\b[^)]*\)\s*\.transform\s*\(',
        r'\.groupby\s*\([^)]*date[^)]*\)\s*\.transform\s*\(',
        r'groupby\s*\(\s*["\']date["\']\s*\)\s*\.transform\s*\(',
    ]
    for p in patterns:
        if re.search(p, code, re.IGNORECASE):
            return True
    return False


# ─────────────────────── Signal normalisation ────────────────────────────

def _normalize_signal(signal: pd.Series) -> pd.Series:
    """
    Ensure signal has a plain DatetimeIndex.

    LLM-generated factor.py sometimes produces a signal with a MultiIndex
    (e.g. (datetime, symbol) or (datetime, column_name)).  A MultiIndex
    containing dtype 'M' (datetime64) cannot be passed to pandas reindex
    against a plain DatetimeIndex — it raises:
        ValueError: cannot include dtype 'M' in a buffer

    Fix: detect a MultiIndex, locate the datetime level, aggregate with mean,
    and return a plain-DatetimeIndex Series.
    """
    if not isinstance(signal.index, pd.MultiIndex):
        return signal

    # Find the level(s) whose dtype is datetime-like
    datetime_levels = [
        i for i, level in enumerate(signal.index.levels)
        if level.dtype.kind == "M"
    ]
    if not datetime_levels:
        # No datetime level found — try to flatten by taking the first level
        datetime_levels = [0]

    dt_level = datetime_levels[0]
    aggregated = signal.groupby(level=dt_level).mean()
    aggregated.index.name = signal.index.names[dt_level]
    aggregated.name = signal.name
    return aggregated


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
          1.525 pts = (105 NTD commission + 200 NTD slippage) ÷ 200 NTD/pt

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

    # ── Next-open execution model ──────────────────────────────────────
    # Signal confirmed at close of bar t  → trade executes at open of bar t+1.
    # executed_pos[t] = position held DURING bar t = pos_adj[t-1]
    executed_pos = pos_adj.shift(1).fillna(0.0)

    # Did the position change at the open of bar t?
    prev_executed = executed_pos.shift(1).fillna(0.0)
    entry_bar = (executed_pos != prev_executed)

    # P&L on entry bars: new_pos × (close[t] - open[t])   (entered at open)
    # P&L on hold  bars: old_pos × (close[t] - close[t-1])
    open_to_close = (df["close"] - df["open"])
    close_to_close = df["close"].diff().fillna(0.0)

    hold_pnl = (
        executed_pos * open_to_close  * entry_bar.astype(float)
        + executed_pos * close_to_close * (~entry_bar).astype(float)
    )

    # Transaction cost: paid whenever executed position changes
    pos_chg = (executed_pos - prev_executed)
    cost = pos_chg.abs() * cost_pts

    net_pnl = hold_pnl - cost

    # ── Cumulative metrics ─────────────────────────────────────────────
    cum_pnl = net_pnl.cumsum()
    total_pts = float(cum_pnl.iloc[-1]) if len(cum_pnl) > 0 else 0.0

    n_days = max(1, (df.index[-1] - df.index[0]).days)
    n_years = n_days / 365.0
    annual_return = total_pts / n_years

    # Max drawdown (in index points on cumulative P&L curve)
    max_drawdown = float((cum_pnl - cum_pnl.cummax()).min())

    # Annualised Sharpe — daily equity return basis (industry standard, matches notebook)
    # Aggregate 1-min P&L to calendar-day totals, then compute daily return Sharpe × √252.
    # This gives a fair comparison across strategies with different market-exposure ratios.
    bars_per_year = 300 * 250  # kept for year-by-year calculation below
    daily_pnl = net_pnl.resample("1D").sum()
    daily_std = float(daily_pnl.std())
    sharpe = float(daily_pnl.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    # Trade count: number of position-change events (open+close = 2 events per round-trip)
    n_trades = int((pos_chg.abs() > 0).sum() // 2)

    # ── Session breakdown (daily Sharpe per session) ──────────────────
    def _session_sharpe(mask: pd.Series) -> float:
        daily_s = net_pnl[mask].resample("1D").sum()
        sd = float(daily_s.std())
        return float(daily_s.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0

    day_mask = df["session"] == "day"
    night_mask = df["session"] == "night"
    day_sharpe = _session_sharpe(day_mask)
    night_sharpe = _session_sharpe(night_mask)

    # ── Year-by-year Sharpe (daily basis, consistent with overall Sharpe) ────
    year_sharpes: dict[str, float] = {}
    for yr, grp in daily_pnl.groupby(daily_pnl.index.year):
        sd_yr = float(grp.std())
        if sd_yr > 0:
            year_sharpes[str(yr)] = round(float(grp.mean() / sd_yr * np.sqrt(252)), 3)
        else:
            year_sharpes[str(yr)] = 0.0

    n_positive_years = sum(1 for v in year_sharpes.values() if v > 0)

    return {
        "sharpe": round(sharpe, 3),
        "annual_return_pts": round(annual_return, 1),
        "max_drawdown_pts": round(max_drawdown, 1),
        "n_trades": n_trades,
        "day_sharpe": round(day_sharpe, 3),
        "night_sharpe": round(night_sharpe, 3),
        "year_sharpes": year_sharpes,
        "n_positive_years": n_positive_years,
    }


def format_result(result: dict | None) -> str:
    if result is None:
        return "No result (execution failed)."
    year_str = ""
    if result.get("year_sharpes"):
        parts = [f"{yr}:{v:+.2f}" for yr, v in sorted(result["year_sharpes"].items())]
        year_str = f" | YearSharpes=[{', '.join(parts)}] | PositiveYears={result.get('n_positive_years', '?')}/{len(result['year_sharpes'])}"
    bias_str = " | LOOKAHEAD_BIAS=YES" if result.get("has_lookahead_bias") else ""
    return (
        f"Sharpe={result['sharpe']:.3f} | "
        f"Annual={result['annual_return_pts']:.0f}pts | "
        f"MaxDD={result['max_drawdown_pts']:.0f}pts | "
        f"Trades={result['n_trades']} | "
        f"DaySharpe={result['day_sharpe']:.3f} | "
        f"NightSharpe={result['night_sharpe']:.3f}"
        f"{year_str}{bias_str}"
    )


# ─────────────────────── Runner ──────────────────────────────────────────

class FuturesFactorRunner(CachedRunner[FuturesFactorExperiment]):
    """
    Executes factor code on the full dataset, runs pandas backtest,
    and stores metrics in exp.result.

    No Qlib dependency.
    """

    def get_cache_key(self, exp: FuturesFactorExperiment) -> str:
        """
        Include test period and cost in the cache key so that changing
        test_start / test_end / cost_pts invalidates all cached results.
        """
        from rdagent.oai.llm_utils import md5_hash

        base_key = super().get_cache_key(exp)

        from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING
        settings_suffix = (
            f"|test_start={FUTURES_FACTOR_PROP_SETTING.test_start}"
            f"|test_end={FUTURES_FACTOR_PROP_SETTING.test_end}"
            f"|cost_pts={run_backtest.__defaults__[0]}"  # cost_pts default from signature
        )
        return md5_hash(base_key + settings_suffix)

    @cache_with_pickle(get_cache_key, CachedRunner.assign_cached_result)
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

        # ── Detect look-ahead bias across all workspaces ─────────────
        has_lookahead = False
        for ws in exp.sub_workspace_list:
            if ws is not None:
                code = getattr(ws, "file_dict", {}).get("factor.py", "")
                if code and _detect_lookahead_bias(code):
                    has_lookahead = True
                    logger.warning("Look-ahead bias detected in factor code (groupby date + transform).")
                    break

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
            signal = _normalize_signal(signal)
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

        result["has_lookahead_bias"] = has_lookahead
        logger.info(f"Backtest result: {format_result(result)}")
        exp.result = result
        return exp
