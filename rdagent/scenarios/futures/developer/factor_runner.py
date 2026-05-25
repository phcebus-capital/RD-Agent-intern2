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

_LOOKAHEAD_SYSTEM_PROMPT = """\
You are an expert quantitative researcher reviewing 1-minute futures factor code
for look-ahead bias.

**Definition.** A factor has look-ahead bias iff the signal value at bar t
depends on data from any bar t' > t. Otherwise it is safe.

**Method.** Trace the data flow before deciding. Find where `signal` (or the
final saved Series) gets its values, then walk backwards through every variable
that feeds it. For each line that involves groupby / map / cumulative ops /
shifts, ask: at bar t, can this expression read from a bar t' > t?

**Key nuance — shift chains are protective.**
A `series.groupby(key).agg()` followed by `key.map(result)` is NOT automatically
biased. Check whether `key` and `result` are both shift(1)-protected so that
each bar only ever sees values derived from strictly earlier bars. If either
the key OR the values are taken from the past (`.shift(1)`, `.cummax()` etc.),
the pattern is safe even though it looks like a full-day broadcast.

**Decision rule.** Default to `has_bias = false`. Only flag `true` if you can
name the SPECIFIC line of code that reads future data. Cite that line in
`reason`. If a suspicious-looking pattern turns out to be shift-protected,
explicitly say so in `reason`.

Reply with JSON only:
{"has_bias": true|false, "reason": "<cite the specific offending line OR \
explain why the suspicious pattern is shift-protected>"}
"""


def _detect_lookahead_bias(code: str) -> bool:
    """
    Use an LLM to detect look-ahead bias in factor code.
    Returns False (assume safe) if the LLM call fails.
    """
    try:
        import json
        from rdagent.oai.llm_utils import APIBackend

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=f"Review this factor code for look-ahead bias:\n\n```python\n{code}\n```",
            system_prompt=_LOOKAHEAD_SYSTEM_PROMPT,
            json_mode=True,
        )
        result = json.loads(response)
        has_bias = bool(result.get("has_bias", False))
        reason   = result.get("reason", "")
        if has_bias:
            logger.warning(f"LLM detected look-ahead bias: {reason}")
        return has_bias

    except Exception as e:
        logger.warning(f"LLM look-ahead check failed ({e}); assuming no bias.")
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
# Single source of truth: delegate to the reference backtest_engiene.
# Two engines giving different results caused weeks of bad evaluation
# (factors with positive reported Sharpe but no actual edge). Avoid that
# by always using the same engine that's used in the manual notebook.

from rdagent.scenarios.futures.backtest_engiene import (
    BacktestConfig as _BTConfig,
    run_backtest as _engine_bt,
    calc_metrics as _engine_metrics,
)


def run_backtest(df: pd.DataFrame, signal: pd.Series, cost_pts: float = 1.525) -> dict:
    """
    Run the reference backtest_engiene and return metrics in factor_runner's
    expected dict format.

    cost_pts (single-side, in index points) maps to backtest_engiene config:
      cost_pts = (fee_per_side + slippage_ticks * tick_value) / point_value
    Default 1.525 pts = (105 + 200) / 200 → fee=105, slip=1×200, point_value=200.
    """
    BacktestConfig, calc_metrics = _BTConfig, _engine_metrics

    # Back-derive fee from cost_pts assuming slippage = 1 tick × 200 NTD = 200 NTD.
    cost_per_side_twd = cost_pts * 200.0
    fee_per_side = max(0, int(round(cost_per_side_twd - 200.0)))

    cfg = BacktestConfig(
        point_value=200,
        fee_per_side=fee_per_side,
        slippage_ticks=1,
        tick_value=200,
        contracts=1,
        execution="next_open",
        stop_loss_points=None,
        take_profit_points=None,
        max_hold_bars=None,
        session_filter=None,
        init_capital=1_000_000,
    )

    # Match the notebook's behaviour: pass the raw combined signal through.
    # The engine casts each bar with `int(sig_arr[i])`, so fractional votes
    # (e.g. 1-of-8 sub-tasks agreeing → 0.125) are truncated to flat. Applying
    # `sign()` here would instead treat any non-zero vote as a full ±1 entry,
    # inflating trade count and diverging from the notebook backtest.
    sig = signal.reindex(df.index).fillna(0)

    trades_df, equity = _engine_bt(df, sig, cfg)

    n_days = max(1, (df.index[-1] - df.index[0]).days)
    if len(trades_df) < 1:
        return {
            "sharpe": 0.0, "annual_return_pts": 0.0,
            "max_drawdown_pts": 0.0, "n_trades": 0,
            "day_sharpe": 0.0, "night_sharpe": 0.0,
            "year_sharpes": {}, "n_positive_years": 0,
        }

    m = calc_metrics(trades_df, equity, cfg)

    # PnL converted to points (equity is cumulative TWD; point_value = 200 NTD/pt)
    cum_pnl_pts = equity / cfg.point_value
    maxdd_pts = float((cum_pnl_pts - cum_pnl_pts.cummax()).min())
    annual_pts = m["total_pnl_twd"] / (n_days / 365.0) / cfg.point_value

    # Year-by-year Sharpe from trade-level daily PnL
    daily_pnl = trades_df.set_index("exit_time")["pnl_twd"].resample("1D").sum()
    year_sharpes: dict[str, float] = {}
    for yr, grp in daily_pnl.groupby(daily_pnl.index.year):
        sd = float(grp.std())
        year_sharpes[str(yr)] = round(float(grp.mean() / sd * np.sqrt(252)), 3) if sd > 0 else 0.0
    n_positive_years = sum(1 for v in year_sharpes.values() if v > 0)

    # Day / night Sharpe: classify each trade by its exit_time session
    def _split_sharpe(mask: pd.Series) -> float:
        sub = trades_df.loc[mask]
        if len(sub) == 0:
            return 0.0
        d = sub.set_index("exit_time")["pnl_twd"].resample("1D").sum()
        sd = float(d.std())
        return round(float(d.mean() / sd * np.sqrt(252)), 3) if sd > 0 else 0.0

    if "session" in df.columns:
        session_at_exit = df["session"].reindex(trades_df["exit_time"]).values
        day_sharpe   = _split_sharpe(session_at_exit == "day")
        night_sharpe = _split_sharpe(session_at_exit == "night")
    else:
        day_sharpe = night_sharpe = 0.0

    return {
        "sharpe"            : round(m["sharpe"], 3),
        "annual_return_pts" : round(annual_pts, 1),
        "max_drawdown_pts"  : round(maxdd_pts, 1),
        "n_trades"          : int(m["total_trades"]),
        "day_sharpe"        : day_sharpe,
        "night_sharpe"      : night_sharpe,
        "year_sharpes"      : year_sharpes,
        "n_positive_years"  : n_positive_years,
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
        Cache key includes:
          - task metadata (from super)
          - test period / cost (changing these invalidates)
          - factor.py contents of every sub_workspace (so a regenerated
            factor.py with the same task info but different code gets a
            fresh cache entry — previously this caused stale metrics to
            be reported when the workspace's result.h5 was overwritten
            by a later run with the same MD5-hashed code dir)
          - baseline workspace code (so changing baseline_factor.py
            invalidates downstream caches that compared against it)
        """
        from rdagent.oai.llm_utils import md5_hash

        base_key = super().get_cache_key(exp)

        from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING
        settings_suffix = (
            f"|test_start={FUTURES_FACTOR_PROP_SETTING.test_start}"
            f"|test_end={FUTURES_FACTOR_PROP_SETTING.test_end}"
            f"|cost_pts={run_backtest.__defaults__[0]}"
            f"|pipeline_v=2"  # v2 = raw signal (notebook-aligned); v1 = sign-cast
        )

        code_chunks: list[str] = []
        for ws in exp.sub_workspace_list:
            if ws is not None:
                code = getattr(ws, "file_dict", {}).get("factor.py", "")
                code_chunks.append(code)
        for based_exp in exp.based_experiments:
            for ws in based_exp.sub_workspace_list:
                if ws is not None:
                    code = getattr(ws, "file_dict", {}).get("factor.py", "")
                    code_chunks.append(code)
        code_suffix = "|code=" + md5_hash("\n---\n".join(code_chunks))

        return md5_hash(base_key + settings_suffix + code_suffix)

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
        # Skip for user-provided baseline (hypothesis is None) — the user is
        # responsible for their own baseline code correctness.
        has_lookahead = False
        if exp.hypothesis is not None:
            for ws in exp.sub_workspace_list:
                if ws is not None:
                    code = getattr(ws, "file_dict", {}).get("factor.py", "")
                    if code and _detect_lookahead_bias(code):
                        has_lookahead = True
                        logger.warning("LLM detected look-ahead bias in factor code.")
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
