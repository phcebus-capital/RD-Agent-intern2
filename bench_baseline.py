"""
Bench the existing A + B-override baseline under the NEW (notebook-aligned)
pipeline so we can update run.sh's strategy prompt with the real SOTA numbers.

What changed: factor_runner.py no longer applies sign() to the combined signal
(see commit). A single hand-coded baseline signal that's already in {-1, 0, +1}
should be unaffected, but we re-measure to be certain.

Splits match conf.py defaults:
  train: 2020-01-01 ~ 2021-12-31
  test : 2022-01-01 ~ 2026-04-15  (notebook calls this "test")
  full : 2020-01-01 ~ 2026-04-15
"""
from __future__ import annotations
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/intern2/workspace/RD-Agent-intern2")
sys.path.insert(0, str(ROOT))

from backtest_engiene import BacktestConfig, run_backtest, calc_metrics  # noqa: E402

BASELINE = ROOT / "git_ignore_folder" / "baseline_factor.py"
DATA_PQT = ROOT / "git_ignore_folder" / "futures_source_data" / "data.parquet"


def exec_baseline() -> pd.Series:
    """Run baseline_factor.py in a temp dir with data.parquet symlinked in."""
    workdir = Path(tempfile.mkdtemp(prefix="baseline_bench_"))
    try:
        (workdir / "data.parquet").symlink_to(DATA_PQT)
        (workdir / "factor.py").write_text(BASELINE.read_text())
        cwd = Path.cwd()
        import os
        os.chdir(workdir)
        try:
            runpy.run_path("factor.py", run_name="__main__")
        finally:
            os.chdir(cwd)
        sig = pd.read_hdf(workdir / "result.h5")
        if isinstance(sig, pd.DataFrame):
            sig = sig.squeeze()
        return sig
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def cfg() -> BacktestConfig:
    return BacktestConfig(
        point_value=200, fee_per_side=105, slippage_ticks=1, tick_value=200,
        contracts=1, execution="next_open",
        stop_loss_points=None, take_profit_points=None, max_hold_bars=None,
        session_filter=None, init_capital=1_000_000,
    )


def metrics_for(df: pd.DataFrame, sig: pd.Series, label: str) -> dict:
    c = cfg()
    trades, eq = run_backtest(df, sig, c)
    m = calc_metrics(trades, eq, c)
    annual_pts = m["total_pnl_twd"] / max(1, (df.index[-1]-df.index[0]).days) * 365 / 200
    print(f"\n=== {label}: {df.index[0].date()} ~ {df.index[-1].date()} "
          f"({len(df):,} bars, {(sig != 0).sum():,} active sig bars) ===")
    print(f"  trades        : {m['total_trades']}")
    print(f"  win_rate      : {m['win_rate']:.1f}%")
    print(f"  profit_factor : {m['profit_factor']:.3f}")
    print(f"  payoff_ratio  : {m['payoff_ratio']:.3f}")
    print(f"  total_pnl_twd : {m['total_pnl_twd']:+,.0f}")
    print(f"  annual_pts    : {annual_pts:+.1f} pts/yr")
    print(f"  sharpe        : {m['sharpe']:+.3f}")
    print(f"  cagr          : {m['cagr']*100:+.2f}%")
    print(f"  max_drawdown  : {m['max_drawdown']*100:+.2f}%")
    print(f"  calmar        : {m['calmar']:+.3f}")
    m["annual_pts"] = annual_pts
    return m


def main() -> None:
    print("Executing baseline_factor.py …")
    sig = exec_baseline()
    print(f"  signal: len={len(sig):,}  dtype={sig.dtype}  uniq={sig.nunique()}")
    print(f"  value_counts: {sig.value_counts().to_dict()}")
    print(f"  range: {sig.index.min()} ~ {sig.index.max()}")

    print("\nLoading data.parquet …")
    df_full = pd.read_parquet(DATA_PQT)

    # Splits (matches future_test.ipynb's train_end = 2022-01-01 implicit)
    train = df_full.loc["2020-01-01":"2021-12-31"]
    test  = df_full.loc["2022-01-01":"2026-04-15"]
    full  = df_full.loc["2020-01-01":"2026-04-15"]

    m_tr = metrics_for(train, sig, "TRAIN  (2020-2021)")
    m_te = metrics_for(test,  sig, "TEST   (2022-2026)")
    m_fl = metrics_for(full,  sig, "FULL   (2020-2026)")

    print("\n" + "=" * 72)
    print(f"{'split':<8}  {'trades':>8}  {'win%':>6}  {'PF':>6}  "
          f"{'Sharpe':>8}  {'CAGR':>8}  {'MaxDD':>9}  {'annual_pts':>12}")
    print("-" * 72)
    for label, m in [("Train", m_tr), ("Test", m_te), ("Full", m_fl)]:
        print(f"{label:<8}  {m['total_trades']:>8d}  {m['win_rate']:>5.1f}%  "
              f"{m['profit_factor']:>6.2f}  {m['sharpe']:>+8.3f}  "
              f"{m['cagr']*100:>+7.2f}%  {m['max_drawdown']*100:>+8.2f}%  "
              f"{m['annual_pts']:>+12.1f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
