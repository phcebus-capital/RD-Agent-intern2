"""
Unit tests for the pure-pandas backtest engine and result formatter.

Covers:
  - run_backtest : vectorised P&L, session-boundary forcing, cost handling,
                   Sharpe computation, day/night split
  - format_result: None sentinel, well-formed dict, output string contents
"""

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.futures.developer.factor_runner import format_result, run_backtest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_ohlcv(
    n: int = 60,
    session: str = "day",
    price_start: float = 18000.0,
    price_step: float = 0.0,
    date: str = "2023-01-03",
) -> pd.DataFrame:
    """Minimal 1-min OHLCV DataFrame for a single session on a single date."""
    start_hm = "08:45" if session == "day" else "15:00"
    idx = pd.date_range(f"{date} {start_hm}", periods=n, freq="1min")
    close = (
        np.full(n, price_start)
        if price_step == 0.0
        else np.arange(price_start, price_start + n * price_step, price_step)[:n]
    )
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 100.0, "session": session},
        index=idx,
    )


def _flat(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index, name="signal")


def _long(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index, name="signal")


def _short(df: pd.DataFrame) -> pd.Series:
    return pd.Series(-1.0, index=df.index, name="signal")


# ── run_backtest ──────────────────────────────────────────────────────────────


class TestRunBacktestOutputShape:
    def test_returns_all_required_keys(self):
        df = _make_ohlcv()
        result = run_backtest(df, _flat(df))
        assert set(result) == {"sharpe", "annual_return_pts", "max_drawdown_pts", "n_trades", "day_sharpe", "night_sharpe"}

    def test_values_are_scalars(self):
        df = _make_ohlcv()
        result = run_backtest(df, _long(df))
        for key, val in result.items():
            assert isinstance(val, (int, float)), f"{key} is not a scalar: {type(val)}"


class TestRunBacktestFlatSignal:
    def test_flat_signal_zero_trades(self):
        df = _make_ohlcv()
        assert run_backtest(df, _flat(df))["n_trades"] == 0

    def test_flat_signal_zero_sharpe(self):
        df = _make_ohlcv()
        assert run_backtest(df, _flat(df))["sharpe"] == 0.0

    def test_flat_signal_zero_annual_return(self):
        df = _make_ohlcv()
        assert run_backtest(df, _flat(df), cost_pts=0.0)["annual_return_pts"] == pytest.approx(0.0, abs=0.1)


class TestRunBacktestConstantPrice:
    """Holding a position against flat prices yields zero holding P&L."""

    def test_long_constant_price_zero_return(self):
        df = _make_ohlcv(price_step=0.0)
        result = run_backtest(df, _long(df), cost_pts=0.0)
        assert result["annual_return_pts"] == pytest.approx(0.0, abs=0.1)

    def test_short_constant_price_zero_return(self):
        df = _make_ohlcv(price_step=0.0)
        result = run_backtest(df, _short(df), cost_pts=0.0)
        assert result["annual_return_pts"] == pytest.approx(0.0, abs=0.1)

    def test_zero_std_gives_zero_sharpe(self):
        df = _make_ohlcv(price_step=0.0)
        result = run_backtest(df, _flat(df), cost_pts=0.0)
        assert result["sharpe"] == 0.0


class TestRunBacktestTrend:
    """Directional edge: correct position + trending price → positive return."""

    def test_long_rising_positive_return(self):
        df = _make_ohlcv(price_step=1.0)
        assert run_backtest(df, _long(df), cost_pts=0.0)["annual_return_pts"] > 0

    def test_short_falling_positive_return(self):
        df = _make_ohlcv(price_step=-1.0)
        assert run_backtest(df, _short(df), cost_pts=0.0)["annual_return_pts"] > 0

    def test_long_falling_negative_return(self):
        df = _make_ohlcv(price_step=-1.0)
        assert run_backtest(df, _long(df), cost_pts=0.0)["annual_return_pts"] < 0

    def test_short_rising_negative_return(self):
        df = _make_ohlcv(price_step=1.0)
        assert run_backtest(df, _short(df), cost_pts=0.0)["annual_return_pts"] < 0


class TestRunBacktestSessionBoundary:
    """Position is forced flat at the last bar of each (date, session) group."""

    def test_single_session_one_roundtrip(self):
        """
        Always-long signal in one session: enters at bar 0, forced out at the
        last bar → exactly 1 completed round-trip.
        """
        df = _make_ohlcv(n=60)
        result = run_backtest(df, _long(df), cost_pts=0.0)
        assert result["n_trades"] == 1

    def test_two_sessions_two_roundtrips(self):
        """Two separate sessions each produce one round-trip."""
        day = _make_ohlcv(n=30, session="day", date="2023-01-03")
        night = _make_ohlcv(n=30, session="night", date="2023-01-03")
        df = pd.concat([day, night]).sort_index()
        sig = pd.Series(1.0, index=df.index, name="signal")
        result = run_backtest(df, sig, cost_pts=0.0)
        assert result["n_trades"] == 2


class TestRunBacktestCost:
    def test_higher_cost_lower_return(self):
        df = _make_ohlcv(n=60, price_step=1.0)
        sig = _long(df)
        r_free = run_backtest(df, sig, cost_pts=0.0)
        r_costly = run_backtest(df, sig, cost_pts=10.0)
        assert r_free["annual_return_pts"] > r_costly["annual_return_pts"]

    def test_zero_cost_baseline(self):
        """Zero cost means P&L equals holding gain only."""
        df = _make_ohlcv(n=60, price_step=1.0)
        result = run_backtest(df, _flat(df), cost_pts=0.0)
        assert result["annual_return_pts"] == pytest.approx(0.0, abs=0.1)


class TestRunBacktestDrawdown:
    def test_drawdown_non_positive(self):
        """Max drawdown is always <= 0 (it measures peak-to-trough loss)."""
        for step in (1.0, -1.0, 0.0):
            df = _make_ohlcv(price_step=step)
            result = run_backtest(df, _long(df))
            assert result["max_drawdown_pts"] <= 0.0, f"Positive drawdown for step={step}"

    def test_monotone_rising_long_has_small_drawdown(self):
        """Perfectly trending long position should have a drawdown close to 0."""
        df = _make_ohlcv(n=100, price_step=1.0)
        result = run_backtest(df, _long(df), cost_pts=0.0)
        assert result["max_drawdown_pts"] == pytest.approx(0.0, abs=1.0)


class TestRunBacktestSessionSplit:
    def test_day_only_data_night_sharpe_is_zero(self):
        df = _make_ohlcv(n=60, session="day", price_step=1.0)
        result = run_backtest(df, _long(df), cost_pts=0.0)
        assert result["night_sharpe"] == 0.0

    def test_night_only_data_day_sharpe_is_zero(self):
        df = _make_ohlcv(n=60, session="night", price_step=1.0)
        result = run_backtest(df, _long(df), cost_pts=0.0)
        assert result["day_sharpe"] == 0.0

    def test_day_session_profitable_night_session_losing(self):
        """
        Long + rising day, long + falling night → day_sharpe > 0, night_sharpe < 0.
        """
        day = _make_ohlcv(n=60, session="day", date="2023-01-03", price_step=1.0)
        night = _make_ohlcv(n=60, session="night", date="2023-01-03", price_step=-1.0)
        df = pd.concat([day, night]).sort_index()
        sig = pd.Series(1.0, index=df.index, name="signal")
        result = run_backtest(df, sig, cost_pts=0.0)
        assert result["day_sharpe"] > 0
        assert result["night_sharpe"] < 0


# ── format_result ─────────────────────────────────────────────────────────────


class TestFormatResult:
    SAMPLE = {
        "sharpe": 1.5,
        "annual_return_pts": 1200.0,
        "max_drawdown_pts": -300.0,
        "n_trades": 500,
        "day_sharpe": 1.2,
        "night_sharpe": 0.8,
    }

    def test_none_returns_failure_sentinel(self):
        assert format_result(None) == "No result (execution failed)."

    def test_returns_string(self):
        assert isinstance(format_result(self.SAMPLE), str)

    def test_sharpe_in_output(self):
        assert "Sharpe=1.500" in format_result(self.SAMPLE)

    def test_annual_return_in_output(self):
        assert "Annual=1200pts" in format_result(self.SAMPLE)

    def test_max_drawdown_in_output(self):
        assert "MaxDD=-300pts" in format_result(self.SAMPLE)

    def test_trades_in_output(self):
        assert "Trades=500" in format_result(self.SAMPLE)

    def test_day_sharpe_in_output(self):
        assert "DaySharpe=1.200" in format_result(self.SAMPLE)

    def test_night_sharpe_in_output(self):
        assert "NightSharpe=0.800" in format_result(self.SAMPLE)

    def test_zero_result_formats_without_error(self):
        zeros = {k: 0 for k in self.SAMPLE}
        s = format_result(zeros)
        assert "Sharpe=0.000" in s
