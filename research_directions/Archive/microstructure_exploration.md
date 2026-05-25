# 目標
在 1-min TX OHLCV 資料上找出 Sharpe > 0.5、年化 trades 介於 400-1500、
扣 1.525 點摩擦成本後仍正報酬的訊號。

# 已飽和方向（這些不要再生成，前 130 個試作 Sharpe 全 < 0）
過去試作徹底探索以下組合，全部失敗，視為失敗模板，請主動避開：
- EMA / SMA 動量交叉（短 vs 長 span 差）
- VWAP deviation Z-score
- Volatility-normalized momentum（rv 當分母）
- Hysteresis / consistency lock / deadband 過濾層
- Realized volatility regime gating（vol_short vs vol_long ratio）
- Capital flow Z-score（volume × signed return 累加）
- 上述元素疊加組成的「regime + lock + deadband」三明治結構

# 鼓勵的探索方向（仍未充分嘗試，在 OHLCV 範圍內可實作）
1. Microstructure proxies
   - Tick-rule signed volume（close vs prev_close 方向 × volume 累加）
   - Amihud illiquidity ( |return| / volume )，反映 informed flow
   - Close 在 bar high-low range 的相對位置（buying pressure indicator）
2. Event-driven anomalies
   - Gap 事件（夜盤收 → 日盤開）後 N 棒的 continuation vs reversal 不對稱
   - Volume spike (≥ 95th pct of rolling window) 後一棒方向 bias
   - 結算日（週三 12:30-13:00）/ 換倉日（週四 08:45-09:00）的特殊 pattern
3. Path-dependent features
   - Local drawdown from rolling peak / runup from rolling trough
   - Time-since-last-extremum
   - Bar count since N-sigma move
4. Higher-order statistics
   - Rolling skewness / kurtosis of 1-min returns
   - Realized semi-variance（上行 vs 下行波動分離）
5. Structural change detection
   - CUSUM 偵測 mean shift 後的 directional bias
   - Range expansion vs contraction phase transition

# 硬性約束
- 年化 trades ∈ [400, 1500]
  （< 400 由 outlier 主導，OOS 不可靠；> 1500 摩擦成本吃光）
- 扣 1.525 點/單摩擦成本後 Sharpe > 0.3
- 時間禁區（多單不可進場，可平倉）：
  - 11:00-12:00, 12:30-13:00（午休流動性枯竭）
  - 週一/三 12:00-12:30（結算日洗盤）
  - 週四 08:45-09:00（結算後換倉）
  - 夜盤全部排除（只做日盤）

# 反例檢測
若新因子核心邏輯（去除參數差異）與最近 5 個試作的「概念骨架」相同
（例：都是「某 momentum + 某 vol gate + 某 lock」），視為無進步，請改方向。
