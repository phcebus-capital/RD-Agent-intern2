"""
Taiwan stock market data generator using FinLab API.

Produces the same HDF5 MultiIndex format as the original Qlib cn_data generator:
  - Index: MultiIndex (datetime, instrument)
  - Columns: $open, $close, $high, $low, $volume, $factor
  - HDF5 key: "data"

Usage:
    # Set your FinLab API token first:
    export FINLAB_API_TOKEN="your_token_here"
    python generate_tw.py

FinLab free tier limitations:
    - Data may be capped at a rolling window (e.g. 3 years)
    - Full history requires a paid subscription
    - Check https://finlab.tw for current plan details
"""

import os
from pathlib import Path

import finlab
import numpy as np
import pandas as pd
from finlab import data


def login() -> None:
    api_token = os.environ.get("FINLAB_API_TOKEN", "")
    if not api_token:
        raise EnvironmentError(
            "FINLAB_API_TOKEN environment variable is not set.\n"
            "Get your token from https://finlab.tw and run:\n"
            "  export FINLAB_API_TOKEN='your_token_here'"
        )
    finlab.login(api_token)


def fetch_raw_data() -> tuple:
    """
    Fetch OHLCV, adjusted close, market cap, and institutional flow from FinLab.

    Returns wide-format DataFrames: index=date, columns=stock_code (e.g. "2330", "2317").

    FinLab field reference:
        price:開盤價      → open price (TWD)
        price:收盤價      → close price (TWD)
        price:最高價      → high price (TWD)
        price:最低價      → low price (TWD)
        price:成交股數    → volume (unit: 股)
        etl:adj_close     → cumulative adjusted close (dividends + splits)
        etl:market_value  → market cap (TWD); available from 2013-04-19 onward
                            NaN ratio ~20% (newly listed / suspended stocks)
        institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)
                          → foreign institutional net buy/sell (shares); $foreign_net
        institutional_investors_trading_summary:投信買賣超股數
                          → trust fund net buy/sell (shares); $trust_net
        institutional_investors_trading_summary:自營商買賣超股數(避險)
                          → dealer hedging net buy/sell (shares); $dealer_net
    """
    print("  Fetching open price...")
    open_ = data.get("price:開盤價")
    print("  Fetching close price...")
    close = data.get("price:收盤價")
    print("  Fetching high price...")
    high = data.get("price:最高價")
    print("  Fetching low price...")
    low = data.get("price:最低價")
    print("  Fetching volume...")
    volume = data.get("price:成交股數")
    print("  Fetching adjusted close price...")
    adj_close = data.get("etl:adj_close")
    print("  Fetching market cap...")
    market_cap = data.get("etl:market_value")
    print("  Fetching foreign institutional net buy/sell...")
    foreign_net = data.get("institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)")
    print("  Fetching trust fund net buy/sell...")
    trust_net = data.get("institutional_investors_trading_summary:投信買賣超股數")
    print("  Fetching dealer hedging net buy/sell...")
    dealer_net = data.get("institutional_investors_trading_summary:自營商買賣超股數(避險)")
    return open_, close, high, low, volume, adj_close, market_cap, foreign_net, trust_net, dealer_net


def build_multiindex_df(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    adj_close: pd.DataFrame,
    market_cap: pd.DataFrame,
    foreign_net: pd.DataFrame,
    trust_net: pd.DataFrame,
    dealer_net: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    instruments: list | None = None,
) -> pd.DataFrame:
    """
    Convert FinLab wide-format DataFrames to Qlib-compatible MultiIndex DataFrame.

    Parameters
    ----------
    start_date : str, optional
        Inclusive start date, e.g. "2010-01-01"
    end_date : str, optional
        Inclusive end date, e.g. "2023-12-31"
    instruments : list[str], optional
        Stock codes to include. If None, includes all available stocks.

    Returns
    -------
    pd.DataFrame
        MultiIndex (datetime, instrument) with columns:
        [$open, $close, $high, $low, $volume, $factor, $market_cap,
         $foreign_net, $trust_net, $dealer_net]

    Notes
    -----
    $factor      = adj_close / close  (復權因子, same convention as Qlib cn_data)
    $market_cap  = market capitalization in TWD (etl:market_value)
                   Available from 2013-04-19; ~20% NaN for newly listed / suspended stocks.
    $foreign_net = foreign institutional net buy/sell shares (signed, unit: 股)
    $trust_net   = trust fund net buy/sell shares (signed, unit: 股)
    $dealer_net  = dealer hedging net buy/sell shares (signed, unit: 股)
    Rows where all price columns are NaN are dropped (suspended / not yet listed).
    """
    # --- Date filtering ---
    dfs = [open_, close, high, low, volume, adj_close, market_cap,
           foreign_net, trust_net, dealer_net]
    if start_date:
        dfs = [df.loc[start_date:] for df in dfs]
    if end_date:
        dfs = [df.loc[:end_date] for df in dfs]
    open_, close, high, low, volume, adj_close, market_cap, foreign_net, trust_net, dealer_net = dfs

    # --- Instrument filtering ---
    if instruments is not None:
        available = close.columns.tolist()
        instruments = [i for i in instruments if i in available]
        open_, close, high, low, volume, adj_close = (
            df[instruments] for df in [open_, close, high, low, volume, adj_close]
        )
        # Sparse columns: reindex to match close universe
        market_cap    = market_cap.reindex(columns=instruments)
        foreign_net   = foreign_net.reindex(columns=instruments)
        trust_net     = trust_net.reindex(columns=instruments)
        dealer_net    = dealer_net.reindex(columns=instruments)

    # --- Compute adjustment factor ---
    # $factor = adj_close / close; avoid inf where close is 0 or NaN
    factor = adj_close.div(close)
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.where(factor.notna(), other=1.0)

    # --- Stack each DataFrame to (datetime, instrument) Series ---
    def _stack(df: pd.DataFrame, col_name: str) -> pd.Series:
        s = df.stack()  # pandas ≥ 1.x compatible
        s.index.names = ["datetime", "instrument"]
        s.name = col_name
        return s

    def _stack_sparse(df: pd.DataFrame, col_name: str) -> pd.Series:
        """Reindex to close index/columns before stacking (handles coverage gaps)."""
        aligned = df.reindex(index=close.index, columns=close.columns)
        return _stack(aligned, col_name)

    combined = pd.concat(
        [
            _stack(open_, "$open"),
            _stack(close, "$close"),
            _stack(high, "$high"),
            _stack(low, "$low"),
            _stack(volume, "$volume"),
            _stack(factor, "$factor"),
            _stack_sparse(market_cap,  "$market_cap"),
            _stack_sparse(foreign_net, "$foreign_net"),
            _stack_sparse(trust_net,   "$trust_net"),
            _stack_sparse(dealer_net,  "$dealer_net"),
        ],
        axis=1,
    )

    # --- Drop rows where all price columns are NaN (stock suspended or not listed yet) ---
    price_cols = ["$open", "$close", "$high", "$low"]
    combined = combined.dropna(subset=price_cols, how="all")

    # FinLab index is already datetime64[ns] date-only (no time component).
    return combined.sort_index()


def select_debug_instruments(close: pd.DataFrame, start: str, end: str, n: int = 100) -> list:
    """
    Select the n most actively traded stocks in [start, end] for the debug dataset.
    Uses number of non-NaN trading days as the proxy for liquidity/stability.
    """
    return (
        close.loc[start:end]
        .notna()
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .index
        .tolist()
    )


def dump_qlib_data(
    df: pd.DataFrame,
    output_dir: str | Path = "~/.qlib/qlib_data/tw_data",
    market_name: str = "tw50",
    n_market: int = 50,
    benchmark_instrument: str = "0050",
    benchmark_alias: str = "TW0050",
) -> None:
    """
    Convert a MultiIndex (datetime, instrument) DataFrame to Qlib's binary provider format.

    Directory layout created:
        output_dir/
            calendars/day.txt                       — one trading date per line
            instruments/all.txt                     — all instruments with date ranges
            instruments/{market_name}.txt           — top-n instruments by trading days
            features/{instrument}/{field}.day.bin   — float32 binary data

    Binary file format (same as Qlib cn_data):
        [start_calendar_idx: float32, value0: float32, value1: float32, ...]
        NaN is written as float32 NaN for missing days within the date range.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex (datetime, instrument) with columns $open $close $high $low
        $volume $factor [$market_cap ...].
    output_dir : str or Path
        Target Qlib provider directory (will be created if absent).
    market_name : str
        Name of the market subset written to instruments/{market_name}.txt.
    n_market : int
        Number of instruments in the market subset (picked by most trading days).
    benchmark_instrument : str
        FinLab instrument code for the index ETF (e.g. "0050" for TW50 ETF).
    benchmark_alias : str
        Alias name used in YAML benchmark field (e.g. "TW0050").
        Its feature files are duplicated under features/{benchmark_alias}/.
    """
    from pathlib import Path as _Path

    output_dir = _Path(output_dir).expanduser()

    # ------------------------------------------------------------------ #
    # 1. Calendar                                                          #
    # ------------------------------------------------------------------ #
    all_dates = sorted(df.index.get_level_values("datetime").unique())
    date_to_idx: dict = {d: i for i, d in enumerate(all_dates)}

    cal_dir = output_dir / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    with open(cal_dir / "day.txt", "w") as f:
        for d in all_dates:
            f.write(pd.Timestamp(d).strftime("%Y-%m-%d") + "\n")
    print(f"  Calendar: {len(all_dates)} trading days  "
          f"({pd.Timestamp(all_dates[0]).date()} → {pd.Timestamp(all_dates[-1]).date()})")

    # ------------------------------------------------------------------ #
    # 2. Instruments                                                       #
    # ------------------------------------------------------------------ #
    inst_dir = output_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)

    # Date range per instrument
    inst_info: dict = {}
    for inst, grp in df.groupby(level="instrument"):
        dates = grp.index.get_level_values("datetime")
        inst_info[inst] = (pd.Timestamp(dates.min()), pd.Timestamp(dates.max()))

    def _fmt_date(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%d")

    # all.txt
    with open(inst_dir / "all.txt", "w") as f:
        for inst in sorted(inst_info):
            start, end = inst_info[inst]
            f.write(f"{inst}\t{_fmt_date(start)}\t{_fmt_date(end)}\n")

    # {market_name}.txt — top-n by number of non-NaN close days
    top_insts = (
        df["$close"]
        .groupby(level="instrument")
        .count()
        .nlargest(n_market)
        .index.tolist()
    )
    with open(inst_dir / f"{market_name}.txt", "w") as f:
        for inst in sorted(top_insts):
            start, end = inst_info[inst]
            f.write(f"{inst}\t{_fmt_date(start)}\t{_fmt_date(end)}\n")
    print(f"  Instruments: {len(inst_info)} total, {n_market} in '{market_name}'")

    # ------------------------------------------------------------------ #
    # 3. Feature binary files                                              #
    # ------------------------------------------------------------------ #
    # Map DataFrame columns → Qlib field names (filename without .day.bin)
    field_map = {
        "$open":        "open",
        "$close":       "close",
        "$high":        "high",
        "$low":         "low",
        "$volume":      "volume",
        "$factor":      "factor",
        "$market_cap":  "market_cap",
        "$foreign_net": "foreign_net",
        "$trust_net":   "trust_net",
        "$dealer_net":  "dealer_net",
    }
    available_fields = {col: name for col, name in field_map.items() if col in df.columns}

    feat_dir = output_dir / "features"
    all_instruments = df.index.get_level_values("instrument").unique().tolist()

    for i, inst in enumerate(all_instruments):
        inst_feat_dir = feat_dir / inst.lower()
        inst_feat_dir.mkdir(parents=True, exist_ok=True)

        inst_series = df.xs(inst, level="instrument")

        for col, field_name in available_fields.items():
            series = inst_series[col]
            non_nan = series.dropna()
            if non_nan.empty:
                continue

            first_date = pd.Timestamp(non_nan.index[0])
            last_date = pd.Timestamp(series.index[-1])
            start_idx = date_to_idx[first_date]
            end_idx = date_to_idx[last_date]

            # Fill the continuous calendar range [start_idx, end_idx]
            full_dates = all_dates[start_idx: end_idx + 1]
            values = series.reindex(full_dates).values  # NaN where no data

            bin_path = inst_feat_dir / f"{field_name}.day.bin"
            with open(bin_path, "wb") as fp:
                np.array([start_idx], dtype="<f4").tofile(fp)
                values.astype("<f4").tofile(fp)

        if (i + 1) % 200 == 0 or (i + 1) == len(all_instruments):
            print(f"  Features written: {i + 1}/{len(all_instruments)} instruments")

    # ------------------------------------------------------------------ #
    # 4. Benchmark alias (e.g. TW0050 → 0050)                             #
    # ------------------------------------------------------------------ #
    if benchmark_instrument in inst_info and benchmark_alias:
        src_dir = feat_dir / benchmark_instrument.lower()
        alias_dir = feat_dir / benchmark_alias.lower()
        if src_dir.exists() and not alias_dir.exists():
            import shutil
            shutil.copytree(src_dir, alias_dir)
            # Also add to instruments/all.txt
            start, end = inst_info[benchmark_instrument]
            with open(inst_dir / "all.txt", "a") as f:
                f.write(f"{benchmark_alias}\t{_fmt_date(start)}\t{_fmt_date(end)}\n")
            print(f"  Benchmark alias: {benchmark_alias} → {benchmark_instrument}")

    print(f"\n  Qlib tw_data written to: {output_dir}")


def main() -> None:
    print("Logging in to FinLab...")
    login()

    print("Fetching raw data from FinLab...")
    open_, close, high, low, volume, adj_close, market_cap, foreign_net, trust_net, dealer_net = fetch_raw_data()

    # ------------------------------------------------------------------ #
    # Full dataset — adjust start_date to match your subscription depth   #
    # Free tier: ~3 years; paid tier: from ~2000                          #
    # ------------------------------------------------------------------ #
    print("\nBuilding full dataset (start: 2010-01-01)...")
    df_all = build_multiindex_df(
        open_, close, high, low, volume, adj_close, market_cap,
        foreign_net, trust_net, dealer_net,
        start_date="2010-01-01",
    )
    df_all.to_hdf("./daily_pv_all.h5", key="data")
    instruments_all = df_all.index.get_level_values("instrument").unique().tolist()
    print(
        f"  Saved daily_pv_all.h5  |  "
        f"{df_all.index.get_level_values('datetime').min().date()} → "
        f"{df_all.index.get_level_values('datetime').max().date()}  |  "
        f"{len(instruments_all)} instruments  |  {len(df_all):,} rows"
    )

    # ------------------------------------------------------------------ #
    # Debug dataset — recent 2 years, top-100 liquid stocks               #
    # ------------------------------------------------------------------ #
    print("\nBuilding debug dataset (2020-01-01 ~ 2021-12-31, 100 instruments)...")
    debug_instruments = select_debug_instruments(close, "2020-01-01", "2021-12-31", n=100)
    df_debug = build_multiindex_df(
        open_, close, high, low, volume, adj_close, market_cap,
        foreign_net, trust_net, dealer_net,
        start_date="2020-01-01",
        end_date="2021-12-31",
        instruments=debug_instruments,
    )
    df_debug.to_hdf("./daily_pv_debug.h5", key="data")
    print(
        f"  Saved daily_pv_debug.h5  |  100 instruments  |  {len(df_debug):,} rows"
    )

    # ------------------------------------------------------------------ #
    # Qlib binary provider data for tw_data                               #
    # ------------------------------------------------------------------ #
    print("\nDumping Qlib provider data to ~/.qlib/qlib_data/tw_data/ ...")
    dump_qlib_data(df_all)

    print("\nDone. Output files:")
    print("  ./daily_pv_all.h5            — full history, all instruments (factor coding)")
    print("  ./daily_pv_debug.h5          — 2020-2021, top-100 instruments (factor coding)")
    print("  ~/.qlib/qlib_data/tw_data/   — Qlib binary provider (backtest / qrun)")


if __name__ == "__main__":
    main()
