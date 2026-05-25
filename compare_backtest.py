"""
Diagnostic: run an RD-Agent factor's signal through BOTH backtest pipelines
(RD-Agent factor_runner style vs. user notebook style) and report where
the metrics diverge.

Pipelines being compared:
  A) RD-Agent (factor_runner.py path):
       df  = read_parquet(git_ignore_folder/futures_source_data/data.parquet)
       sig = mean(workspace_signals).reindex(df.index).fillna(0).apply(sign)
       run_backtest(df.loc[test_start:test_end], sig.loc[...])

  B) Notebook (future_test.ipynb style):
       df  = concat(KBar_1m_ML/*.csv).set_index(datetime).sort_index()
       sig = mean(workspace_signals)            # no reindex, no sign
       run_backtest(df.loc[2020-01-01:2026-04-15], sig.loc[...])
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/intern2/workspace/RD-Agent-intern2")
sys.path.insert(0, str(ROOT))

from backtest_engiene import BacktestConfig, run_backtest, calc_metrics  # noqa: E402

# ── Pick the factor result we want to reproduce ─────────────────────────
WS_HASHES = [
    "34d638911c3f4e1893f3e021e43da7e5",
    "d2a621a4b4204166a471c236d94ec5a5",
    "d2427fa6a9f648ba8ac91ab9691a7953",
    "1525aba196164113a50d8f7679d2cd37",
    "f337aba423024368af98560f05a9b620",
    "1f0eef4b0dfa46f1a5922b54b00645fe",
    "9368530693464a6688f939933ffc9541",
    "fe183efcedd94cb985bc402db655d89f",
]
WS_BASE = ROOT / "git_ignore_folder" / "RD-Agent_workspace"

TEST_START = "2020-01-01"
TEST_END = "2026-04-15"


def load_signals() -> list[pd.Series]:
    out = []
    for h in WS_HASHES:
        p = WS_BASE / h / "result.h5"
        if not p.exists():
            print(f"  ! missing {p}")
            continue
        s = pd.read_hdf(p)
        if isinstance(s, pd.DataFrame):
            s = s.squeeze()
        # mimic _normalize_signal: collapse MultiIndex if any
        if isinstance(s.index, pd.MultiIndex):
            dt_levels = [i for i, lv in enumerate(s.index.levels) if lv.dtype.kind == "M"]
            lvl = dt_levels[0] if dt_levels else 0
            s = s.groupby(level=lvl).mean()
        out.append(s)
    return out


def combine(signals: list[pd.Series]) -> pd.Series:
    combined = pd.concat(signals, axis=1).mean(axis=1)
    combined.name = "signal"
    return combined


def load_df_parquet() -> pd.DataFrame:
    p = ROOT / "git_ignore_folder" / "futures_source_data" / "data.parquet"
    return pd.read_parquet(p)


def load_df_csv() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "KBar_1m_ML/KBar_1m_ML" / "*.csv")))
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


def cfg() -> BacktestConfig:
    return BacktestConfig(
        point_value=200,
        fee_per_side=105,
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


def fmt_metrics(m: dict) -> str:
    if not m:
        return "(no trades)"
    return (
        f"trades={m['total_trades']:>5}  "
        f"win%={m['win_rate']:>5.1f}  "
        f"PF={m['profit_factor']:>5.2f}  "
        f"Sharpe={m['sharpe']:>+6.3f}  "
        f"PnL={m['total_pnl_twd']:>+14,.0f}  "
        f"MDD={m['max_drawdown']*100:>+7.2f}%"
    )


def main() -> None:
    print("Loading workspace signals …")
    sigs = load_signals()
    print(f"  Got {len(sigs)} signals; lengths: {[len(s) for s in sigs]}")
    print(f"  Per-signal value_counts (first 3 ws):")
    for i, s in enumerate(sigs[:3]):
        vc = s.value_counts(dropna=False).head(5).to_dict()
        print(f"    ws[{i}]: {vc}")

    combined_raw = combine(sigs)
    print(f"\nCombined signal: len={len(combined_raw)}  dtype={combined_raw.dtype}")
    print(f"  value_counts (top 10): {combined_raw.value_counts().head(10).to_dict()}")
    print(f"  non-zero fraction: {(combined_raw != 0).mean()*100:.2f}%")

    print("\nLoading both DataFrames …")
    df_pq = load_df_parquet()
    df_cs = load_df_csv()
    print(f"  parquet (RD-Agent): len={len(df_pq):,}  range={df_pq.index.min()} ~ {df_pq.index.max()}")
    print(f"  CSV     (notebook): len={len(df_cs):,}  range={df_cs.index.min()} ~ {df_cs.index.max()}")
    aligned = df_pq.index.equals(df_cs.index)
    print(f"  index equal? {aligned}")
    if not aligned:
        diff_pq_only = df_pq.index.difference(df_cs.index)
        diff_cs_only = df_cs.index.difference(df_pq.index)
        print(f"    bars only in parquet: {len(diff_pq_only):,}  only in CSV: {len(diff_cs_only):,}")
        common = df_pq.index.intersection(df_cs.index)
        if len(common) > 0:
            ohlc_diff = (df_pq.loc[common, ["open","high","low","close"]] - df_cs.loc[common, ["open","high","low","close"]]).abs().sum().sum()
            print(f"    OHLC abs diff on common bars: {ohlc_diff:.4f}")

    c = cfg()

    # ── Pipeline A: RD-Agent factor_runner style ────────────────────────
    df_A = df_pq.loc[TEST_START:TEST_END]
    sig_A = combined_raw.reindex(df_A.index).fillna(0)
    sig_A = sig_A.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    print("\n[A] RD-Agent pipeline (parquet df + reindex + sign):")
    print(f"    df_A: {len(df_A):,} bars   sig_A non-zero: {(sig_A != 0).sum():,}")
    trades_A, eq_A = run_backtest(df_A, sig_A, c)
    m_A = calc_metrics(trades_A, eq_A, c)
    print("   ", fmt_metrics(m_A))

    # ── Pipeline B: Notebook style ─────────────────────────────────────
    df_B = df_cs.loc[TEST_START:TEST_END]
    sig_B = combined_raw.loc[TEST_START:TEST_END]
    print("\n[B] Notebook pipeline (CSV df + raw signal, NO sign):")
    print(f"    df_B: {len(df_B):,} bars   sig_B non-zero: {(sig_B != 0).sum():,}   unique: {sig_B.nunique()}")
    trades_B, eq_B = run_backtest(df_B, sig_B, c)
    m_B = calc_metrics(trades_B, eq_B, c)
    print("   ", fmt_metrics(m_B))

    # ── Pipeline C: Notebook df + RD-Agent's sign step ─────────────────
    # Isolate "is the difference from the signal continuous→sign cast?"
    sig_C = sig_B.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    print("\n[C] CSV df + sign(combined) (isolate sign-cast effect):")
    print(f"    sig_C non-zero: {(sig_C != 0).sum():,}")
    trades_C, eq_C = run_backtest(df_B, sig_C, c)
    m_C = calc_metrics(trades_C, eq_C, c)
    print("   ", fmt_metrics(m_C))

    # ── Pipeline D: parquet df + raw signal (no sign) ──────────────────
    # Isolate "is it the data?"
    sig_D = combined_raw.reindex(df_A.index).fillna(0)
    print("\n[D] Parquet df + raw signal, NO sign (isolate data-source effect):")
    trades_D, eq_D = run_backtest(df_A, sig_D, c)
    m_D = calc_metrics(trades_D, eq_D, c)
    print("   ", fmt_metrics(m_D))

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  A  RD-Agent (parquet + reindex + sign) : {fmt_metrics(m_A)}")
    print(f"  B  Notebook (CSV + raw)                : {fmt_metrics(m_B)}")
    print(f"  C  CSV + sign                          : {fmt_metrics(m_C)}")
    print(f"  D  Parquet + raw                       : {fmt_metrics(m_D)}")
    print("=" * 70)
    print("Interpretation:")
    print("  A vs C : isolates df parquet vs CSV (and reindex/fillna).")
    print("  A vs D : isolates the sign-cast step (raw mean vs ±1).")
    print("  B vs C : isolates the sign-cast step on the CSV df.")
    print("  B vs D : isolates the data source with raw signal.")


if __name__ == "__main__":
    main()
