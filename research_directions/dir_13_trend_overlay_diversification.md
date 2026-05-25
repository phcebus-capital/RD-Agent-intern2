# 方向 13：TSMOM Overlay — 並列日線時序動能 sleeve 做組合層分散

**目標**：在 BT7（單帳戶 cross-side reactive，日級 Sharpe ~2.76 / MDD −158K / 7-0 正報酬）之外**並列**一條日線時序動能 sleeve，以小風險權重組合，提升逐年穩定與抗 OOS 衰減，而非改動 BT7 訊號本身。

## 核心想法

```python
trend_t = sign( mean_L( close[t-1] - close[t-1-L] ) )      # L ∈ {10,20,40,60} ensemble
trend_pnl_t = trend_t * (close[t] - close[t-1]) * MULT - turnover_cost
book = bt7_pnl + W * trend_pnl                              # W ≈ 0.1~0.2（風險偏好旋鈕）
```

Moskowitz, Ooi & Pedersen (2012) *Time Series Momentum*：過去 1–12 月報酬正→未來續漲，且**對股市左尾為正**——正是 BT7 偏多/做頭流血時缺的方向。

## 為什麼是「組合層」不是「訊號層」

- 不碰 BT1/BT2/cross-side composer，不新增日內進場特徵 → 零洩漏、trades 恆 ≤614。
- 加的是**獨立 edge**（趨勢 premium），不是調舊 edge → 比參數掃描 robust，且是 BT7 in-sample Sharpe OOS 衰減時的保險。

## 實測已知（2026-05-22 fast-check，務必當已知前提，別重新發現）

- BT7 + 0.2·TREND：淨利 +500K、**6/7 年勝 BT7**、TREND 扛 2022 空頭 +132K / 2020 +124K。
- **但日 Sharpe 持平（2.76→2.73）、MDD 惡化（−158K→−196K）**。月相關 −0.34（負）vs 日相關 +0.11（正）= 頻率陷阱。
- 數學鐵律：SR_trend≈0.5 ≪ SR_BT7≈2.76 → 組合 Sharpe 上限 √(2.76²+0.5²)≈2.78，**動不了 headline**。

## 強制方向

- 風險加權，**禁止等口數**（TREND 連續曝險 vol 遠大於選擇性 BT7，等權會被它主導）。
- 第一輪只用 ensemble-sign TSMOM（rule of three）；Donchian 與 TSMOM 相關 0.74–0.82（冗餘），要混就 vol-balanced 成一條 TREND。
- 價值定位在「**逐年穩定 + OOS 保險**」，**不是**拉 Sharpe / 壓 MDD。把目標函數設成「總報酬↑ 且 逐年不更差 且 MDD 不退步太多」，別優化 headline Sharpe。

## 已飽和 / 請避開

- 別期待 Sharpe 或 MDD 大幅改善——基底太強，數學上限已知（+~2%）。
- 別對 BT7 自身做 vol-targeting / equity-vol / dd-mask 風控（三 factory 已證無效，是 reactive 死路）。
- 別把 w 調到 >0.4 硬拉淨利（MDD 線性惡化、net/|MDD| 崩）。

## 硬約束

- 全扣摩擦 COST_RT=210、MULT=200、slippage 可加 1–2 tick。
- trades 上限 1000（天然滿足）、下限維持 BT7 的 614（overlay 不砍）。
- **OOS / walk-forward 必跑**：負月相關子期不穩（−0.234 / −0.012 / −0.407），in-sample 的 L、w 不可直接信。

## 自我檢查清單

1. 組合改善是否通過 BT7 top-5 outlier 移除後仍成立？（baseline PF 由少數 outlier 主導）
2. net/|MDD| 退步 → 對照「固定加 beta 的 const baseline」，確認是分散不是純加槓桿。
3. 負月相關是否單一 regime 造成？split-sample / per-year 穩定性。
4. TSMOM premium 在 walk-forward OOS 是否仍正？崩了就釘死「TX 內無可用分散 sleeve、BT7 單獨上線」。
5. 若要兼顧壓 MDD → 與方向 12（prospective 隔夜 gate）串聯，別單靠本 overlay。

## 文獻

- Moskowitz, Ooi & Pedersen (2012) *Time Series Momentum*, JFE 104(2):228–250
- Hurst, Ooi & Pedersen (2017) *A Century of Evidence on Trend-Following Investing*, AQR
- Harvey et al. (2018) *The Impact of Volatility Targeting*, JPM — 趨勢策略左尾修正
- López de Prado (2018) *AFML* Ch11-12 — walk-forward / 過擬合防護
