"""
Prepare 1-min TX futures data for use by the RD-Agent futures scenario.

Reads per-day CSVs from tw-futures-data/data_clean/kbar_1min_clean/ and
writes two parquet files that factor.py scripts will load:

  git_ignore_folder/futures_source_data/data.parquet       ← full dataset
  git_ignore_folder/futures_source_data_debug/data.parquet ← debug (first N days)

Run once before starting the RD-Agent loop:
  python rdagent/scenarios/futures/prepare_data.py

Optional arguments:
  --kbar_dir PATH     Source CSV directory  (default: tw-futures-data/data_clean/kbar_1min_clean)
  --out_dir PATH      Full-data output dir  (default: git_ignore_folder/futures_source_data)
  --debug_dir PATH    Debug-data output dir (default: git_ignore_folder/futures_source_data_debug)
  --debug_days INT    Number of trading days in debug dataset (default: 60)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_kbar_csvs(kbar_dir: Path) -> pd.DataFrame:
    files = sorted(kbar_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {kbar_dir}")

    print(f"  Loading {len(files)} daily CSV files from {kbar_dir} ...")
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["datetime"], encoding="utf-8-sig")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("datetime").reset_index(drop=True)

    # Set DatetimeIndex
    merged = merged.set_index("datetime")
    merged.index = pd.DatetimeIndex(merged.index)

    # Keep only needed columns
    cols = ["open", "high", "low", "close", "volume", "session"]
    merged = merged[cols]

    print(
        f"  Total bars: {len(merged):,}  "
        f"({merged.index[0].date()} → {merged.index[-1].date()})"
    )
    return merged


def save_parquet(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    df.to_parquet(out_path, engine="pyarrow")
    print(f"  [{label}] Saved {len(df):,} bars → {out_path}")


def main(
    kbar_dir: str = "tw-futures-data/data_clean/kbar_1min_clean",
    out_dir: str = "git_ignore_folder/futures_source_data",
    debug_dir: str = "git_ignore_folder/futures_source_data_debug",
    debug_days: int = 60,
) -> None:
    kbar_path = Path(kbar_dir)
    out_path = Path(out_dir)
    debug_path = Path(debug_dir)

    print("Loading 1-min K-bar data ...")
    df = load_kbar_csvs(kbar_path)

    print("\nWriting full dataset ...")
    save_parquet(df, out_path, "full")

    print("\nWriting debug dataset ...")
    # Take the first `debug_days` unique trading dates
    dates = sorted(df.index.normalize().unique())
    cutoff_dates = dates[:debug_days]
    cutoff_ts = pd.Timestamp(cutoff_dates[-1]) + pd.Timedelta(days=1)
    df_debug = df[df.index < cutoff_ts]
    save_parquet(df_debug, debug_path, f"debug ({debug_days} days)")

    print("\nDone.")
    print(f"  Full  : {out_path / 'data.parquet'}")
    print(f"  Debug : {debug_path / 'data.parquet'}")
    print("\nSet these env vars before running the RD-Agent futures loop:")
    print(f"  export FUTURES_CoSTEER_data_folder={out_path.resolve()}")
    print(f"  export FUTURES_CoSTEER_data_folder_debug={debug_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TX 1-min futures data for RD-Agent.")
    parser.add_argument("--kbar_dir", default="tw-futures-data/data_clean/kbar_1min_clean")
    parser.add_argument("--out_dir", default="git_ignore_folder/futures_source_data")
    parser.add_argument("--debug_dir", default="git_ignore_folder/futures_source_data_debug")
    parser.add_argument("--debug_days", type=int, default=60)
    args = parser.parse_args()
    main(args.kbar_dir, args.out_dir, args.debug_dir, args.debug_days)
