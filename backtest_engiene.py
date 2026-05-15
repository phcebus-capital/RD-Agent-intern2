"""
Taiwan Futures (TX) Backtesting Engine
台灣期貨（台指期）回測引擎

支援多空雙向 1-min K bar-by-bar 回測，適用於 TX 近月合約。
"""

import numpy as np
import pandas as pd

# ─── 合約規格 ─────────────────────────────────────────────────────────────────
TX_POINT_VALUE  = 200   # 每點 200 TWD（大台）
TX_TICK_SIZE    = 1     # 最小跳動 1 點
TX_TICK_VALUE   = 200   # 最小跳動 200 TWD


# ─── 回測設定 ─────────────────────────────────────────────────────────────────
class BacktestConfig:
    """
    回測參數設定。

    Attributes
    ----------
    point_value        : 每點TWD（大台=200, 小台=50）
    fee_per_side       : 單邊手續費 TWD/口
    slippage_ticks     : 滑價檔數（每檔 = tick_value）
    tick_value         : 每檔 TWD
    contracts          : 固定口數
    execution          : 'current_close' | 'next_open'
                         current_close = 訊號K棒收盤成交（當根）
                         next_open     = 下一根開盤成交（T+1，較保守）
    stop_loss_points   : 停損點數（None = 不使用）
    take_profit_points : 停利點數（None = 不使用）
    max_hold_bars      : 最大持倉根數（None = 不限）
    session_filter     : 'day' | 'night' | None（None = 日夜全時段）
    init_capital       : 初始資金 TWD（用於淨值/CAGR計算）
    """

    def __init__(
        self,
        point_value: float = TX_POINT_VALUE,
        fee_per_side: float = 105,
        slippage_ticks: int = 1,
        tick_value: float = TX_TICK_VALUE,
        contracts: int = 1,
        execution: str = "next_open",
        stop_loss_points: float | None = None,
        take_profit_points: float | None = None,
        max_hold_bars: int | None = None,
        session_filter: str | None = None,
        init_capital: float = 1_000_000,
    ):
        self.point_value        = point_value
        self.fee_per_side       = fee_per_side
        self.slippage_twd       = slippage_ticks * tick_value
        self.cost_per_side      = fee_per_side + self.slippage_twd
        self.cost_roundtrip     = self.cost_per_side * 2
        self.contracts          = contracts
        self.execution          = execution
        self.stop_loss_points   = stop_loss_points
        self.take_profit_points = take_profit_points
        self.max_hold_bars      = max_hold_bars
        self.session_filter     = session_filter
        self.init_capital       = init_capital

    def __repr__(self) -> str:
        return (
            f"BacktestConfig("
            f"execution={self.execution!r}, "
            f"cost_rt={self.cost_roundtrip:.0f} TWD/口, "
            f"contracts={self.contracts}, "
            f"sl={self.stop_loss_points}, tp={self.take_profit_points}, "
            f"session={self.session_filter!r})"
        )


# ─── 核心回測引擎 ─────────────────────────────────────────────────────────────
def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    執行期貨回測。

    Parameters
    ----------
    df     : OHLCV DataFrame，index 為 datetime，需包含 open/high/low/close/volume，
             選用 session 欄（'day'/'night'）。
    signal : pd.Series (index 與 df 相同)，值為 -1（空）/ 0（無）/ +1（多）。
    config : BacktestConfig，預設用標準TX設定。

    Returns
    -------
    trades_df    : 每筆交易明細 DataFrame
    equity_curve : 累計損益（TWD）pd.Series，index 與 df 相同
    """
    if config is None:
        config = BacktestConfig()

    sig = signal.reindex(df.index).fillna(0)

    # session 篩選：將非目標時段的訊號清零
    if config.session_filter and "session" in df.columns:
        sig = sig.where(df["session"] == config.session_filter, 0)

    # 轉成 numpy 加速
    sig_arr  = sig.values.astype(np.float32)
    opens    = df["open"].values.astype(np.float64)
    closes   = df["close"].values.astype(np.float64)
    highs    = df["high"].values.astype(np.float64)
    lows     = df["low"].values.astype(np.float64)
    ts       = df.index
    sessions = df["session"].values if "session" in df.columns else np.full(len(df), "all")

    n = len(df)
    records = []

    # 持倉狀態
    position    : int   = 0     # 0=flat, 1=long, -1=short
    entry_bar   : int   = -1
    entry_price : float = 0.0

    def _entry_px(bar_idx: int) -> float:
        if config.execution == "next_open":
            return opens[bar_idx] if bar_idx < n else closes[bar_idx - 1]
        return closes[bar_idx]

    def _exit_px(bar_idx: int) -> float:
        if config.execution == "next_open":
            return opens[bar_idx] if bar_idx < n else closes[bar_idx - 1]
        return closes[bar_idx]

    def _open_trade(bar_idx: int, direction: int) -> None:
        nonlocal position, entry_bar, entry_price
        position    = direction
        entry_bar   = bar_idx
        entry_price = _entry_px(bar_idx)

    def _close_trade(bar_idx: int, reason: str) -> dict:
        ep  = _exit_px(bar_idx)
        pnl_pts = (ep - entry_price) * position
        pnl_twd = (
            pnl_pts * config.point_value * config.contracts
            - config.cost_roundtrip * config.contracts
        )
        return {
            "entry_time"  : ts[entry_bar],
            "entry_price" : entry_price,
            "exit_time"   : ts[bar_idx],
            "exit_price"  : ep,
            "direction"   : "多" if position == 1 else "空",
            "hold_bars"   : bar_idx - entry_bar,
            "pnl_points"  : pnl_pts,
            "gross_pnl_twd": pnl_pts * config.point_value * config.contracts,
            "cost_twd"    : config.cost_roundtrip * config.contracts,
            "pnl_twd"     : pnl_twd,
            "exit_reason" : reason,
            "session"     : sessions[entry_bar],
        }

    for i in range(n):
        s     = int(sig_arr[i])
        s_prv = int(sig_arr[i - 1]) if i > 0 else 0

        if position == 0:
            # ── 進場偵測（訊號邊緣）─────────────────
            if s != 0 and s_prv == 0:
                exec_bar = (i + 1) if config.execution == "next_open" and i + 1 < n else i
                _open_trade(exec_bar, s)
            continue

        # ── 持倉中：逐根檢查出場條件 ─────────────────
        held = i - entry_bar
        if held <= 0:
            continue

        unrealized = (closes[i] - entry_price) * position  # 未實現損益（點）
        should_exit  = False
        exit_reason  = ""

        # 1. 停損
        if config.stop_loss_points and unrealized <= -config.stop_loss_points:
            should_exit = True
            exit_reason = "stop_loss"

        # 2. 停利
        if not should_exit and config.take_profit_points and unrealized >= config.take_profit_points:
            should_exit = True
            exit_reason = "take_profit"

        # 3. 超過最大持倉根數
        if not should_exit and config.max_hold_bars and held >= config.max_hold_bars:
            should_exit = True
            exit_reason = "max_hold"

        # 4. 訊號消失或反轉
        if not should_exit and (s == 0 or s == -position):
            should_exit = True
            exit_reason = "signal_end" if s == 0 else "signal_reverse"

        if should_exit:
            exec_bar = (i + 1) if config.execution == "next_open" and i + 1 < n else i
            if exec_bar >= n:
                exec_bar = n - 1
            rec = _close_trade(exec_bar, exit_reason)
            records.append(rec)
            position = 0

            # 反轉立即進場
            if exit_reason == "signal_reverse" and s != 0:
                _open_trade(exec_bar, s)

    # ── 強制平倉（回測結束）────────────────────────────
    if position != 0:
        rec = _close_trade(n - 1, "liquidation")
        records.append(rec)

    # ── 整理 trades DataFrame ─────────────────────────
    trades_df = pd.DataFrame(records)
    if len(trades_df) == 0:
        return trades_df, pd.Series(0.0, index=df.index, name="equity")

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"]  = pd.to_datetime(trades_df["exit_time"])
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

    # ── 建立按時間軸的累計損益曲線 ────────────────────
    pnl_ts = trades_df.set_index("exit_time")["pnl_twd"]
    pnl_ts = pnl_ts.groupby(pnl_ts.index).sum()          # 同一時間多筆合併
    equity  = pnl_ts.reindex(df.index).fillna(0).cumsum()
    equity.name = "equity"

    return trades_df, equity


# ─── 績效計算 ─────────────────────────────────────────────────────────────────
def calc_metrics(
    trades_df: pd.DataFrame,
    equity: pd.Series,
    config: BacktestConfig | None = None,
) -> dict:
    """
    計算完整回測績效指標。

    Returns
    -------
    dict with keys:
        total_trades, win_rate, avg_win_twd, avg_loss_twd,
        profit_factor, payoff_ratio, total_pnl_twd,
        cagr, sharpe, max_drawdown, calmar
    """
    if config is None:
        config = BacktestConfig()

    if len(trades_df) == 0:
        return {}

    pnl  = trades_df["pnl_twd"]
    wins = pnl[pnl > 0]
    loss = pnl[pnl <= 0]

    win_rate  = len(wins) / len(pnl) * 100
    avg_win   = float(wins.mean())  if len(wins) > 0 else 0.0
    avg_loss  = float(loss.mean())  if len(loss) > 0 else 0.0
    pf        = wins.sum() / abs(loss.sum()) if loss.sum() != 0 else float("inf")
    payoff    = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # 淨值序列
    nav = (equity + config.init_capital) / config.init_capital
    nav = nav[nav > 0]

    # 最大回撤
    peak       = nav.cummax()
    dd_series  = (nav - peak) / peak
    mdd        = float(dd_series.min())

    # Sharpe（日頻化）
    daily_nav = nav.resample("1D").last().dropna()
    daily_ret = daily_nav.pct_change().dropna()
    sharpe    = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if len(daily_ret) > 1 and daily_ret.std() > 0
        else 0.0
    )

    # CAGR
    span_days = (nav.index[-1] - nav.index[0]).days
    total_ret = float(nav.iloc[-1] / nav.iloc[0])
    cagr      = float(total_ret ** (365.25 / span_days) - 1) if span_days > 0 else 0.0
    calmar    = cagr / abs(mdd) if mdd != 0 else float("inf")

    return {
        "total_trades"   : len(trades_df),
        "win_rate"       : win_rate,
        "avg_win_twd"    : avg_win,
        "avg_loss_twd"   : avg_loss,
        "profit_factor"  : pf,
        "payoff_ratio"   : payoff,
        "total_pnl_twd"  : float(pnl.sum()),
        "cagr"           : cagr,
        "sharpe"         : sharpe,
        "max_drawdown"   : mdd,
        "calmar"         : calmar,
    }


def print_report(trades_df: pd.DataFrame, equity: pd.Series, config: BacktestConfig | None = None) -> None:
    """列印回測摘要報表。"""
    if config is None:
        config = BacktestConfig()
    m = calc_metrics(trades_df, equity, config)
    if not m:
        print("無任何交易紀錄。")
        return

    pnl = trades_df["pnl_twd"]

    print("=" * 60)
    print(f"  回測結果摘要 — TX 台指期近月（{config.contracts} 口）")
    print("=" * 60)
    print(f"  成交策略        : {config.execution}")
    print(f"  來回交易成本    : {config.cost_roundtrip:,.0f} TWD/口")
    if config.stop_loss_points:
        print(f"  停損點數        : {config.stop_loss_points} 點")
    if config.take_profit_points:
        print(f"  停利點數        : {config.take_profit_points} 點")
    print()
    print(f"  總交易筆數      : {m['total_trades']:,}")
    print(f"  勝率            : {m['win_rate']:.1f}%")
    print(f"  平均獲利        : {m['avg_win_twd']:,.0f} TWD")
    print(f"  平均虧損        : {m['avg_loss_twd']:,.0f} TWD")
    print(f"  盈虧比          : {m['payoff_ratio']:.2f}")
    print(f"  獲利因子 (PF)   : {m['profit_factor']:.3f}")
    print()
    print(f"  累計淨損益      : {m['total_pnl_twd']:>+14,.0f} TWD")
    print(f"  CAGR            : {m['cagr']*100:+.2f}%")
    print(f"  Sharpe Ratio    : {m['sharpe']:.3f}")
    print(f"  最大回撤        : {m['max_drawdown']*100:.2f}%")
    print(f"  Calmar Ratio    : {m['calmar']:.3f}")
    print("=" * 60)

    # 多空分析
    for direction in ["多", "空"]:
        sub = trades_df[trades_df["direction"] == direction]
        if len(sub) == 0:
            continue
        sub_pnl = sub["pnl_twd"]
        sub_win = sub_pnl[sub_pnl > 0]
        sub_pf  = (sub_win.sum() / abs(sub_pnl[sub_pnl <= 0].sum())
                   if sub_pnl[sub_pnl <= 0].sum() != 0 else float("inf"))
        print(
            f"  {direction}單：{len(sub):>4} 筆  "
            f"勝率 {len(sub_win)/len(sub)*100:.1f}%  "
            f"PF {sub_pf:.2f}  "
            f"淨損益 {sub_pnl.sum():>+12,.0f} TWD"
        )

    # 年度彙整
    print()
    print("  ─ 年度損益 ─")
    trades_df["_year"] = trades_df["exit_time"].dt.year
    annual = (
        trades_df.groupby("_year")["pnl_twd"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "淨損益", "count": "筆數"})
    )
    for yr, row in annual.iterrows():
        print(f"    {yr}  {row['淨損益']:>+12,.0f} TWD  ({row['筆數']:.0f} 筆)")
    trades_df.drop(columns="_year", inplace=True)

    # 出場原因分布
    print()
    print("  ─ 出場原因 ─")
    for reason, cnt in trades_df["exit_reason"].value_counts().items():
        print(f"    {reason:<20s}: {cnt:>4} ({cnt/len(trades_df)*100:.1f}%)")

    # 持倉時間分析
    avg_bars = trades_df["hold_bars"].mean()
    print()
    print(f"  平均持倉根數    : {avg_bars:.1f} 根 (≈{avg_bars:.0f} 分鐘)")
    print("=" * 60)


# ─── 繪圖 ─────────────────────────────────────────────────────────────────────
def plot_report(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | pd.Series | None = None,
    equity: pd.Series | None = None,
    config: BacktestConfig | None = None,
    backend: str = "auto",
    title: str = "TX 台指期 回測結果",
) -> None:
    """
    繪製完整回測圖表（Plotly 優先，fallback matplotlib）。

    Parameters
    ----------
    df           : 原始 OHLCV DataFrame
    trades_df    : run_backtest 回傳的交易明細
    benchmark_df : 基準線 DataFrame（含 出場時間/進場時間/累計獲利金額）
                   或已處理過的 Series（index=時間, values=累計獲利+init_capital）
    equity       : run_backtest 回傳的累計損益序列
    backend      : 'auto' | 'plotly' | 'matplotlib'
    title        : 主標題
    """
    if config is None:
        config = BacktestConfig()

    has_plotly = False
    if backend in ("auto", "plotly"):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            has_plotly = True
        except ImportError:
            pass

    if has_plotly and backend != "matplotlib":
        _plot_plotly(df, trades_df, benchmark_df, equity, config, title)
    else:
        _plot_mpl(df, trades_df, benchmark_df, equity, config, title)


def _plot_plotly(df, trades_df, benchmark_df, equity, config, title):
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    nav = (equity + config.init_capital) / config.init_capital
    dd  = (nav - nav.cummax()) / nav.cummax()

    # ── 圖 1：淨值 + 回撤 ─────────────────────────────────────────────────────
    fig1 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("淨值曲線", "水下曲線（回撤）"),
        vertical_spacing=0.06,
    )
    fig1.add_trace(
        go.Scatter(x=nav.index, y=nav.values, name="淨值",
                   line=dict(color="#2ecc71", width=2)),
        row=1, col=1,
    )
    fig1.add_hline(y=1, line_dash="dot", line_color="gray", row=1, col=1)
    fig1.add_trace(
        go.Scatter(x=dd.index, y=dd.values, name="回撤",
                   fill="tozeroy", line=dict(color="#e74c3c", width=1)),
        row=2, col=1,
    )
    fig1.update_layout(
        template="plotly_white", height=550,
        title=dict(text=f"<b>{title}</b>  淨值與回撤", font=dict(size=15)),
        legend=dict(orientation="h", y=1.04, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=70, b=40),
    )
    
    # Benchmark 累計獲利線
    if benchmark_df is not None and hasattr(benchmark_df, "empty") and not benchmark_df.empty:
        if isinstance(benchmark_df, pd.DataFrame):
            bm = benchmark_df.copy()
            bm["時間"] = pd.to_datetime(bm["出場時間"].fillna(bm["進場時間"]))
            bm["累計獲利金額"] = pd.to_numeric(
                bm["累計獲利金額"].astype(str).str.replace(",", ""),
                errors="coerce"
            )
            bm = bm.dropna(subset=["時間", "累計獲利金額"]).sort_values("時間")
            fig1.add_trace(
                go.Scatter(x=bm["時間"], y=bm["累計獲利金額"] / config.init_capital + 1,
                           name="基準線（累計獲利）", line=dict(color="#3498db", width=2, dash="dash")),
                row=1, col=1,
            )
        elif isinstance(benchmark_df, pd.Series):
            # 已預處理的 Series（values = 累計獲利 + init_capital）
            fig1.add_trace(
                go.Scatter(x=benchmark_df.index, y=benchmark_df.values / config.init_capital,
                           name="基準線（累計獲利）", line=dict(color="#3498db", width=2, dash="dash")),
                row=1, col=1,
            )
            bm_nav = pd.Series(
                pd.to_numeric(benchmark_df, errors="coerce").values / config.init_capital,
                index=pd.to_datetime(benchmark_df.index),
                dtype="float64",
            ).replace([np.inf, -np.inf], np.nan).dropna().sort_index()

            bm_nav = bm_nav[bm_nav > 0]
            if not bm_nav.empty:
                bm_dd = (bm_nav - bm_nav.cummax()) / bm_nav.cummax()
                fig1.add_trace(
                    go.Scatter(
                        x=bm_dd.index,
                        y=bm_dd.values,
                        name="基準線回撤",
                        line=dict(color="#3498db", width=1.5, dash="dash"),
                    ),
                    row=2, col=1,
                )
    fig1.update_yaxes(title_text="淨值", row=1, col=1)
    fig1.update_yaxes(title_text="回撤", tickformat=".1%", row=2, col=1)
    fig1.show()

    # ── 圖 2：進出場標記（最近 N 筆交易的時間範圍）─────────────────────────
    if len(trades_df) >= 1:
        recent_n = min(30, len(trades_df))
        plot_start = trades_df.iloc[-recent_n]["entry_time"]
        plot_end   = trades_df.iloc[-1]["exit_time"]
        plot_df    = df.loc[plot_start:plot_end]
        plot_trd   = trades_df[trades_df["entry_time"] >= plot_start]

        fig2 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25],
            subplot_titles=(f"最近 {recent_n} 筆交易進出場（1分K）", "Volume"),
            vertical_spacing=0.05,
        )
        fig2.add_trace(
            go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"], high=plot_df["high"],
            low=plot_df["low"],   close=plot_df["close"],
            name="K棒",
            increasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
            decreasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
            showlegend=False,
            ),
            row=1, col=1,
        )
        fig2.update_xaxes(
            rangebreaks=[
            dict(bounds=["sat", "mon"]),                 # 週末
            dict(bounds=[13.75, 8.75], pattern="hour"),  # 13:45 ~ 隔日 08:45
            ]
        )
        for _, t in plot_trd.iterrows():
            color = "rgba(46,204,113,0.15)" if t["direction"] == "多" else "rgba(231,76,60,0.15)"
            fig2.add_vrect(x0=t["entry_time"], x1=t["exit_time"],
                           fillcolor=color, layer="below", line_width=0)

        # 多空進出場標記
        long_entries  = plot_trd[plot_trd["direction"] == "多"]
        short_entries = plot_trd[plot_trd["direction"] == "空"]
        price_std = plot_df["close"].std()

        if len(long_entries):
            fig2.add_trace(go.Scatter(
                x=long_entries["entry_time"],
                y=plot_df.reindex(long_entries["entry_time"])["low"],
                mode="markers", name="多進",
                marker=dict(symbol="triangle-up", size=12, color="#2ecc71",
                            line=dict(width=1.5, color="#145a32")),
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(
                x=long_entries["exit_time"],
                y=plot_df.reindex(long_entries["exit_time"])["close"],
                mode="markers", name="多出",
                marker=dict(symbol="x", size=10, color="#27ae60", line=dict(width=2)),
            ), row=1, col=1)

        if len(short_entries):
            fig2.add_trace(go.Scatter(
                x=short_entries["entry_time"],
                y=plot_df.reindex(short_entries["entry_time"])["high"],
                mode="markers", name="空進",
                marker=dict(symbol="triangle-down", size=12, color="#e74c3c",
                            line=dict(width=1.5, color="#7b241c")),
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(
                x=short_entries["exit_time"],
                y=plot_df.reindex(short_entries["exit_time"])["close"],
                mode="markers", name="空出",
                marker=dict(symbol="x", size=10, color="#c0392b", line=dict(width=2)),
            ), row=1, col=1)

        vol_color = [
            "rgba(239,83,80,0.7)" if c >= o else "rgba(38,166,154,0.7)"
            for c, o in zip(plot_df["close"], plot_df["open"])
        ]
        fig2.add_trace(
            go.Bar(x=plot_df.index, y=plot_df["volume"], marker_color=vol_color,
                   showlegend=False),
            row=2, col=1,
        )
        fig2.update_layout(
            template="plotly_white", height=650,
            title=dict(text=f"<b>{title}</b>　進出場標記", font=dict(size=15)),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.04, xanchor="right", x=1),
            margin=dict(l=60, r=20, t=70, b=40),
        )
        fig2.show()

    # ── 圖 3：每筆損益分布 ─────────────────────────────────────────────────────
    fig3 = go.Figure()
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in trades_df["pnl_twd"]]
    fig3.add_trace(go.Bar(
        x=trades_df.index,
        y=trades_df["pnl_twd"],
        marker_color=colors,
        name="每筆損益",
    ))
    fig3.add_hline(y=0, line_dash="dot", line_color="gray")
    fig3.update_layout(
        template="plotly_white", height=350,
        title=dict(text=f"<b>{title}</b>　每筆損益條形圖", font=dict(size=15)),
        xaxis_title="第N筆", yaxis_title="損益 (TWD)",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig3.show()

    # ── 圖 4：年度損益 ─────────────────────────────────────────────────────────
    annual = (
        trades_df.copy()
        .assign(_year=lambda x: x["exit_time"].dt.year)
        .groupby("_year")["pnl_twd"].sum()
    )
    fig4 = go.Figure(go.Bar(
        x=annual.index.astype(str),
        y=annual.values,
        marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in annual.values],
    ))
    fig4.add_hline(y=0, line_dash="dot", line_color="gray")
    fig4.update_layout(
        template="plotly_white", height=350,
        title=dict(text=f"<b>{title}</b>　年度損益", font=dict(size=15)),
        xaxis_title="年份", yaxis_title="損益 (TWD)",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig4.show()


def _plot_mpl(df, trades_df, benchmark_df, equity, config, title):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    nav = (equity + config.init_capital) / config.init_capital
    dd  = (nav - nav.cummax()) / nav.cummax()

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs  = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    # 淨值曲線
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(nav.index, nav.values, color="#2ecc71", linewidth=1.5, label="淨值")
    ax1.axhline(1, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_title(f"{title} — 淨值曲線", fontsize=12)
    ax1.set_ylabel("淨值")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    # 回撤曲線
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dd.index, dd.values, 0, color="#e74c3c", alpha=0.4)
    ax2.plot(dd.index, dd.values, color="#e74c3c", linewidth=0.8)
    ax2.set_ylabel("回撤")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax2.grid(True, alpha=0.25)
    ax2.set_title("水下曲線（回撤）")

    # 每筆損益
    ax3 = fig.add_subplot(gs[2, 0])
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in trades_df["pnl_twd"]]
    ax3.bar(trades_df.index, trades_df["pnl_twd"], color=colors, width=0.8)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_title("每筆損益")
    ax3.set_xlabel("第N筆")
    ax3.set_ylabel("損益 (TWD)")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # 年度損益
    annual = (
        trades_df.copy()
        .assign(_year=lambda x: x["exit_time"].dt.year)
        .groupby("_year")["pnl_twd"].sum()
    )
    ax4 = fig.add_subplot(gs[2, 1])
    colors_y = ["#2ecc71" if v >= 0 else "#e74c3c" for v in annual.values]
    ax4.bar(annual.index.astype(str), annual.values, color=colors_y)
    ax4.axhline(0, color="black", linewidth=0.5)
    ax4.set_title("年度損益")
    ax4.set_ylabel("損益 (TWD)")
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.savefig("backtest_report.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("已存檔 backtest_report.png")
