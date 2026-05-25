"""
Naive autoresearch loop for TX futures factors.

One LLM call per iteration. Backtest_engiene is the only source of truth.
No staged pipeline, no strategy map, no proposal-coder-runner-feedback layers.

Run:
    python -m rdagent.scenarios.futures.autoresearch --loops 50

Output layout:
    autoresearch_runs/<timestamp>/
        baseline/factor.py + result.h5
        trial_000/factor.py + result.h5 + trial.json
        trial_001/...
        summary.json          # cumulative trial log

FUTURE WORK (not implemented — current loop relies solely on model's
training-time knowledge):
    External context retrieval. The model has no live access to recent
    quant blogs / SSRN / GitHub repos / practitioner notes. arxiv q-fin
    is mostly toy models that don't help. Worth integrating something
    like Tavily/Brave search API or a curated knowledge file ONLY if a
    high-quality source is found. Skipping for now — frontier model's
    own knowledge of VPIN/BOCPD/microstructure is already strong enough
    to iterate on a narrow task.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.futures.backtest_engiene import (
    BacktestConfig,
    calc_metrics,
    run_backtest,
)

DEFAULT_DATA_PATH = "git_ignore_folder/futures_source_data/data.parquet"
DEFAULT_BASELINE = "git_ignore_folder/baseline_factor.py"

SYSTEM_PROMPT = """You design 1-minute TX (Taiwan futures) intraday factors.

Each iteration, you output ONE python file factor.py that:
1. Reads data.parquet (1-min OHLCV, columns: open/high/low/close/volume/session)
2. Computes signal: pd.Series indexed by df.index, values are -1/0/+1 (or any value, np.sign is applied downstream)
3. Saves: signal.to_hdf('result.h5', key='signal', mode='w')

The backtest:
- Edge-triggered: position opens at next bar open when signal changes 0→±1
- Holds until signal goes to 0 or flips sign
- Round-trip cost: 3.05 points (1.525 per side)

Hard rules:
- No look-ahead: signal at bar t must only use data from bars t' ≤ t.
- groupby(date).transform('agg') leaks future-day data. Use .cummax / .cumsum / .expanding.
- df.shift(-N) is forbidden.
- Hold duration should average ≥ 30 bars (otherwise friction wipes alpha).

Output format — EXACTLY this:
# Hypothesis: <one concise sentence describing the signal logic>
<the full python factor.py code starting from imports>

Output ONLY the comment line and code. No prose, no markdown fences, no JSON wrapper."""

USER_TEMPLATE = """## Current SOTA baseline (the bar to beat — do not regenerate this)
{baseline_summary}
Baseline code (for context — your factor must be UNCORRELATED with this):
```python
{baseline_code}
```

## Recent trial results
{trials_log}

## Your task — propose NEW factor for trial {iter_idx}
{scope}

Targets:
- Sharpe > 0.3 after cost
- Annualized trades 200 – 2000
- Positive in ≥ 5 of 7 years (2020-2026)
- |Correlation with baseline| < 0.3

Avoid: tweaking parameters of a recent trial; concept skeletons we already tried (see trial list).

Write factor.py now."""


# ─────────────────────── Trial execution ────────────────────────────────


def execute_factor(
    code: str, work_dir: Path, data_path: Path, timeout: int = 300
) -> tuple[Optional[pd.Series], str]:
    """Run factor.py in `work_dir`, return (signal, stdout_or_error)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "factor.py").write_text(code)

    dst = work_dir / "data.parquet"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(data_path.resolve(), dst)

    try:
        proc = subprocess.run(
            [sys.executable, "factor.py"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT after {timeout}s"

    if proc.returncode != 0:
        out = (proc.stdout + "\n" + proc.stderr).strip()
        return None, f"FAILED (exit {proc.returncode})\n{out[-1500:]}"

    result_path = work_dir / "result.h5"
    if not result_path.exists():
        return None, "result.h5 not produced"

    try:
        return pd.read_hdf(result_path, key="signal"), "ok"
    except Exception as e:
        return None, f"failed to read result.h5: {e}"


def evaluate_signal(signal: pd.Series, df: pd.DataFrame) -> dict:
    """Run backtest_engiene, return metrics dict."""
    sig = signal.reindex(df.index).fillna(0)
    sig = sig.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    cfg = BacktestConfig(
        point_value=200,
        fee_per_side=105,
        slippage_ticks=1,
        tick_value=200,
        contracts=1,
        execution="next_open",
        init_capital=1_000_000,
    )
    trades, equity = run_backtest(df, sig, cfg)

    if len(trades) < 5:
        return {
            "sharpe": 0.0,
            "n_trades": int(len(trades)),
            "active_ratio": round(float((sig != 0).mean()), 4),
            "error": "too_few_trades",
        }

    m = calc_metrics(trades, equity, cfg)
    annual = (
        trades.assign(year=lambda x: x["exit_time"].dt.year)
        .groupby("year")["pnl_twd"]
        .sum()
    )
    year_sharpes = {}
    for yr, grp in trades.assign(year=lambda x: x["exit_time"].dt.year).groupby("year"):
        daily = grp.set_index("exit_time")["pnl_twd"].resample("1D").sum()
        sd = float(daily.std()) if len(daily) > 1 else 0.0
        year_sharpes[str(int(yr))] = (
            round(float(daily.mean() / sd * (252**0.5)), 2) if sd > 0 else 0.0
        )

    return {
        "sharpe": round(m["sharpe"], 3),
        "cagr_pct": round(m["cagr"] * 100, 2),
        "maxdd_pct": round(m["max_drawdown"] * 100, 2),
        "win_rate_pct": round(m["win_rate"], 1),
        "profit_factor": round(m["profit_factor"], 3),
        "n_trades": int(m["total_trades"]),
        "pos_yrs": int((annual > 0).sum()),
        "total_yrs": int(len(annual)),
        "year_sharpes": year_sharpes,
        "active_ratio": round(float((sig != 0).mean()), 4),
        "total_pnl_twd": int(round(m["total_pnl_twd"])),
    }


# ─────────────────────── LLM I/O ────────────────────────────────────────


def parse_llm_output(raw: str) -> tuple[str, str]:
    """Extract (hypothesis, code) from LLM response."""
    text = raw.strip()
    # Strip markdown fences if model insists
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        if text.startswith("python"):
            text = text[6:]
        text = text.lstrip("\n").rstrip().rstrip("`").rstrip()

    lines = text.split("\n")
    hypothesis = ""
    for i, line in enumerate(lines):
        if line.strip().startswith("# Hypothesis:"):
            hypothesis = line.split(":", 1)[1].strip()
            break
    return hypothesis, text


def format_trial_log(trial: dict, max_code_lines: int = 12) -> str:
    m = trial.get("metrics", {})
    err = m.get("error")
    if err:
        line = f"[Trial {trial['idx']}] ERROR: {err}"
    else:
        line = (
            f"[Trial {trial['idx']}] "
            f"Sharpe={m['sharpe']:+.3f} "
            f"Trades={m['n_trades']} "
            f"MaxDD={m['maxdd_pct']:+.1f}% "
            f"PosYrs={m['pos_yrs']}/{m['total_yrs']} "
            f"Corr={trial.get('correlation', 0):+.2f}"
        )
    desc = trial.get("hypothesis", "")
    if desc:
        line += f"\n  Hypothesis: {desc}"
    code_lines = trial.get("code", "").split("\n")[:max_code_lines]
    code_preview = "\n  ".join(code_lines)
    if len(trial.get("code", "").split("\n")) > max_code_lines:
        code_preview += f"\n  # ... ({len(trial['code'].split(chr(10))) - max_code_lines} more lines)"
    return line + f"\n  Code preview:\n  {code_preview}"


# ─────────────────────── Main loop ──────────────────────────────────────


def autoresearch_loop(
    n_iter: int,
    n_recent_trials: int,
    data_path: str,
    baseline_path: str,
    output_dir: str,
    scope: str,
):
    data_p = Path(data_path)
    if not data_p.exists():
        raise FileNotFoundError(f"Data not found: {data_p}")

    out_p = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Output: {out_p}")
    print(f"[setup] Loading data {data_p}...")
    df = pd.read_parquet(data_p)
    print(f"[setup] {len(df):,} bars, {df.index[0]} ~ {df.index[-1]}")

    # ── Baseline ────────────────────────────────────────────────────
    baseline_code = Path(baseline_path).read_text()
    print(f"[setup] Running baseline {baseline_path}...")
    baseline_sig, msg = execute_factor(baseline_code, out_p / "baseline", data_p)
    if baseline_sig is None:
        raise RuntimeError(f"Baseline failed: {msg}")
    bm = evaluate_signal(baseline_sig, df)
    baseline_summary = (
        f"Sharpe={bm['sharpe']:+.3f}, CAGR={bm['cagr_pct']:+.1f}%, "
        f"MaxDD={bm['maxdd_pct']:+.1f}%, Trades={bm['n_trades']}, "
        f"PosYrs={bm['pos_yrs']}/{bm['total_yrs']}"
    )
    print(f"[setup] Baseline: {baseline_summary}\n")
    (out_p / "baseline_metrics.json").write_text(json.dumps(bm, indent=2))

    # ── Iterations ──────────────────────────────────────────────────
    api = APIBackend()
    trials: list[dict] = []
    best_sharpe = bm["sharpe"]

    for iter_idx in range(n_iter):
        print(f"\n{'─' * 70}\n[iter {iter_idx}] generating factor...")

        recent = trials[-n_recent_trials:]
        trials_log = (
            "\n\n".join(format_trial_log(t) for t in recent)
            if recent
            else "(no trials yet — this is the first iteration)"
        )
        user_prompt = USER_TEMPLATE.format(
            iter_idx=iter_idx,
            baseline_summary=baseline_summary,
            baseline_code=baseline_code,
            trials_log=trials_log,
            scope=scope,
        )

        try:
            raw = api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
            )
        except Exception as e:
            print(f"  LLM call failed: {e}")
            continue

        hypothesis, code = parse_llm_output(raw)
        trial_ws = out_p / f"trial_{iter_idx:03d}"
        print(f"  Hypothesis: {hypothesis or '(none extracted)'}")
        print(f"  Executing factor...")

        signal, run_msg = execute_factor(code, trial_ws, data_p)
        trial: dict = {
            "idx": iter_idx,
            "hypothesis": hypothesis,
            "code": code,
            "run_msg": run_msg,
        }

        if signal is None:
            trial["metrics"] = {"error": run_msg[:200]}
            trial["correlation"] = 0.0
            print(f"  Failed: {run_msg[:120]}")
        else:
            metrics = evaluate_signal(signal, df)
            corr = float(
                signal.reindex(baseline_sig.index)
                .fillna(0)
                .corr(baseline_sig.fillna(0))
            )
            trial["metrics"] = metrics
            trial["correlation"] = round(corr, 4)

            if metrics.get("error"):
                print(f"  {metrics['error']}; n_trades={metrics.get('n_trades', 0)}")
            else:
                marker = ""
                if (
                    metrics["sharpe"] > best_sharpe
                    and abs(corr) < 0.3
                    and metrics["pos_yrs"] >= 5
                ):
                    marker = "  ★ NEW BEST"
                    best_sharpe = metrics["sharpe"]
                print(
                    f"  Sharpe={metrics['sharpe']:+.3f} "
                    f"Trades={metrics['n_trades']} "
                    f"MaxDD={metrics['maxdd_pct']:+.1f}% "
                    f"PosYrs={metrics['pos_yrs']}/{metrics['total_yrs']} "
                    f"Corr={corr:+.3f}{marker}"
                )

        (trial_ws / "trial.json").write_text(json.dumps(trial, indent=2, default=str))
        trials.append(trial)

        # Rolling summary
        summary = []
        for t in trials:
            m = t.get("metrics", {})
            summary.append(
                {
                    "idx": t["idx"],
                    "hypothesis": t.get("hypothesis", ""),
                    "sharpe": m.get("sharpe", 0),
                    "n_trades": m.get("n_trades", 0),
                    "pos_yrs": m.get("pos_yrs", 0),
                    "correlation": t.get("correlation", 0),
                    "error": m.get("error", ""),
                }
            )
        (out_p / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'═' * 70}\nDone. {n_iter} iterations. Results in {out_p}")
    return out_p


# ─────────────────────── CLI ────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loops", type=int, default=30, help="number of iterations")
    p.add_argument(
        "--recent",
        type=int,
        default=8,
        help="number of recent trials to feed back to model",
    )
    p.add_argument("--data", default=DEFAULT_DATA_PATH)
    p.add_argument("--baseline", default=DEFAULT_BASELINE)
    p.add_argument("--output", default="autoresearch_runs")
    p.add_argument(
        "--scope",
        default=(
            "Find a factor that complements baseline. "
            "Priority: long-side intraday triggers, pre-09:30 setups, post-13:30 setups, "
            "and strong-uptrend days where baseline is silent. "
            "Avoid: any momentum / VWAP-deviation / ORB / volume-thrust / mean-reversion "
            "skeleton — these are saturated."
        ),
    )
    args = p.parse_args()

    autoresearch_loop(
        n_iter=args.loops,
        n_recent_trials=args.recent,
        data_path=args.data,
        baseline_path=args.baseline,
        output_dir=args.output,
        scope=args.scope,
    )


if __name__ == "__main__":
    main()
