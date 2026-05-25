"""
Verify that factor_runner.run_backtest (post-fix) gives the same metrics as the
notebook's run_backtest call. Uses Loop_15 cached signals.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/intern2/workspace/RD-Agent-intern2")
sys.path.insert(0, str(ROOT))

from backtest_engiene import BacktestConfig, run_backtest as engine_bt, calc_metrics
from rdagent.scenarios.futures.developer.factor_runner import run_backtest as rdagent_bt

WS_HASHES = [
    "34d638911c3f4e1893f3e021e43da7e5", "d2a621a4b4204166a471c236d94ec5a5",
    "d2427fa6a9f648ba8ac91ab9691a7953", "1525aba196164113a50d8f7679d2cd37",
    "f337aba423024368af98560f05a9b620", "1f0eef4b0dfa46f1a5922b54b00645fe",
    "9368530693464a6688f939933ffc9541", "fe183efcedd94cb985bc402db655d89f",
]
WS_BASE = ROOT / "git_ignore_folder" / "RD-Agent_workspace"

sigs = []
for h in WS_HASHES:
    s = pd.read_hdf(WS_BASE / h / "result.h5")
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    sigs.append(s)
combined = pd.concat(sigs, axis=1).mean(axis=1)
combined.name = "signal"

df = pd.read_parquet(ROOT / "git_ignore_folder/futures_source_data/data.parquet")
df = df.loc["2020-01-01":"2026-04-15"]
sig = combined.loc["2020-01-01":"2026-04-15"]

# RD-Agent path (after fix)
r = rdagent_bt(df, sig)

# Notebook path
cfg = BacktestConfig(point_value=200, fee_per_side=105, slippage_ticks=1,
                     tick_value=200, contracts=1, execution="next_open",
                     stop_loss_points=None, take_profit_points=None,
                     max_hold_bars=None, session_filter=None,
                     init_capital=1_000_000)
trades, eq = engine_bt(df, sig, cfg)
m = calc_metrics(trades, eq, cfg)

print(f"{'metric':<14}  {'RD-Agent':>14}  {'notebook':>14}  {'match':>6}")
print("-" * 56)
def row(name, a, b, fmt):
    same = "✓" if abs(a - b) < 1e-6 else "✗"
    print(f"{name:<14}  {fmt.format(a):>14}  {fmt.format(b):>14}  {same:>6}")

row("n_trades",     r["n_trades"],          m["total_trades"],  "{:.0f}")
row("sharpe",       r["sharpe"],            round(m["sharpe"], 3), "{:+.3f}")
row("PnL_pts",      r["annual_return_pts"], round(m["total_pnl_twd"] / 200 / (max(1,(df.index[-1]-df.index[0]).days)/365.0), 1), "{:+.1f}")
row("MaxDD_pts",    r["max_drawdown_pts"],  round((eq/200 - (eq/200).cummax()).min(), 1), "{:+.1f}")

print()
print(f"RD-Agent full result: {r}")
print(f"\nNotebook metrics:")
for k, v in m.items():
    print(f"  {k:18s} = {v}")
