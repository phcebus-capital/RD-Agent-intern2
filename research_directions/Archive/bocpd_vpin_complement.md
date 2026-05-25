# ★★★ 絕對紅線（每一輪都要遵守，不接受任何例外）★★★
新因子 method_type 必須是 bocpd 或 vpin，二選一。
任何「momentum / mean_reversion / VWAP_deviation / ORB / volume_thrust / pin_bar /
regime_filter / HMM / ML / permutation_entropy / OBI」即使包裝成 BOCPD 或 VPIN 都會被拒絕
（看 hypothesis_specification 裡的 STRICTLY PROHIBITED 清單）。
寫 hypothesis 前先問自己：核心觸發機制是不是「偵測 statistical changepoint」或
「估計 informed flow imbalance」？如果答案模糊，不要提出這個 hypothesis。

# 目標
在 1-min TX OHLCV 資料上，找出能與**現有 baseline (B-override + A) 互補**的新因子，
扣 1.525 點/單摩擦成本後 Sharpe > 0.3，年化 trades ∈ [150, 800]。

## 已建立的兩個訊號（嚴禁重複生成同類概念）

### Signal A: blended_anchor_skew_vol（每日開盤前計算，整個日盤期間固定）
- 用 20-bar low/close/high 區間定位當前 close 的相對位置
- 用 60-bar return skew 估計分布偏度（symmetric → regime ok）
- 用 60-bar/200-bar vol ratio 調整訊號強度
- 訊號在 09:15-13:00 期間 active，方向靠 np.sign 取
- **核心概念**：「session 開盤前已知資訊」決定整日方向

### Signal B: ORB short on 08:45-09:30 breakdown
- 08:45-09:30 形成 opening range
- 收盤跌破 OR low + 收盤低於 VWAP + close-position ≤ 0.25 + body ≥ 0.3 + 10 日趨勢非正
- 一旦觸發，做空持有到 13:30 強制平倉
- **核心概念**：「panic-sell breakdown 觸發後持有」

## 合併方式：B-override + A
- 預設用 A 的方向
- 只要 B 觸發，無條件覆蓋為 -1（做空）
- B 結束後（13:30 後）回到 A 方向

# 缺口分析（請填補這些方向）
現有組合的限制：
1. **長部位 alpha 弱**：A 提供方向但變化少，純 long 部分缺乏事件觸發
2. **多頭日表現差**：2023 大多頭年僅 break-even
3. **09:30 之前空白**：早盤前 45 分鐘完全沒有訊號
4. **13:30 之後空白**：午盤後資金閒置
5. **與 B 相關性過高的因子無用**：B 已覆蓋 panic-sell 場景，任何 short-breakdown 類訊號都重複

# 強制要求：新因子核心邏輯必須屬於以下兩個方向之一
**只接受 BOCPD 或 VPIN 概念的因子**。任何其他方向（純動量、純均值回歸、純 ORB、
純 VWAP deviation、ML/HMM 規則）一律拒絕。

## 強制方向 1：結構性變點偵測（BOCPD 概念）
- 核心思路：偵測價格或成交量序列在何時發生統計意義上的均值或波動率跳變
- 以「確認發生變點之後的新狀態」作為進出場依據，而非固定回望窗口的歷史均值
- 進場：偵測到向上跳變（多頭新 regime 建立）後做多
- 出場：偵測到向下跳變或 regime 崩潰時平倉
- 重點：讓「變點發生的時機」本身成為訊號，而不是事後用均線確認
- 數學工具範例：Page-Hinkley CUSUM、Bayesian Online Changepoint Detection、
  run length distribution、posterior probability of changepoint

## 強制方向 2：知情交易壓力（VPIN 概念）
- 核心思路：將成交量依價格方向分桶，估計買方與賣方主動成交的失衡程度
- 失衡偏向買方代表知情資金正在單邊建倉，作為進場觸發或強度過濾
- 進場：買方失衡超過閾值，且失衡狀態持續穩定時做多
- 出場：失衡消退或翻轉為賣方主導時平倉
- 重點：衡量「誰在交易、力道是否集中」而非價格本身的走勢
- 數學工具範例：BVC (Bulk Volume Classification)、Lee-Ready tick rule、
  equal-volume buckets、order flow imbalance, signed volume entropy

# 在強制方向上要填補的缺口（從這些角度切入）

## 缺口 1：日內 long-side 事件觸發（最缺）
- 不要做 ORB long（已驗證 Sharpe 0.06，無效）
- BOCPD 應用：偵測「下跌結束」變點 → long
- VPIN 應用：buy-side stealth accumulation（價格沒漲但 buy imbalance 累積）→ long

## 缺口 2：早盤 08:45-09:30 setup（B 沒覆蓋的時段）
- BOCPD 應用：早盤前 15 分鐘 vol/flow 變點偵測
- VPIN 應用：開盤後 buy/sell imbalance 首次明確偏向某一邊

## 缺口 3：13:30 後尾盤 setup（資金完全閒置）
- BOCPD 應用：13:30 平倉後的 second-wave 變點
- VPIN 應用：尾盤前 institutional rebalancing 偵測

## 缺口 4：與 B 互補的多頭日專用
- 在「10 日趨勢 ≥ +0.5%」的強多頭日才 active
- B 在這些日子被 trend filter 擋掉
- 兩個方向都可：BOCPD 偵測強多頭日的微結構變點，或 VPIN 偵測強買方主導

# 相關性硬性約束（新增）
新因子的 signal series 與 baseline B 的 Pearson 相關性必須 |ρ| < 0.3。
具體做法：
- 計算新因子 signal 在 baseline B 觸發的 bars 上的平均方向
- 若兩者高度同向（或反向）視為冗餘，直接淘汰

# 已飽和、請主動避開的方向
以下方向已被大量試作，概念骨架雷同者一律視為無效，不得再生成：

## 動量類（全部飽和）
- EMA / SMA 動量交叉（短 vs 長 span 差）
- VWAP deviation Z-score（任何形式的收盤 vs VWAP 距離）
- Volatility-normalized momentum（rv 當分母正規化報酬）
- Opening Range Breakout（ORB）：任意時間窗的開盤區間突破
- ATR 倍數追蹤停損搭配動量進場

## 成交量類（全部飽和）
- Capital flow Z-score（volume × signed return 累加後 z-score）
- Volume-weighted thrust / OBI：(close–open)/range × volume 類結構
- Bull/bear volume ratio：多空方向成交量比值或累加差
- 任何以「方向成交量 / 總成交量」為核心的比率指標

## 過濾 / 篩選層（全部飽和）
- Hysteresis / consistency lock / deadband 過濾層
- Realized volatility regime gating（vol_short/vol_long ratio 切換進出場）
- Body ratio 過濾（|close–open|/range ≥ 閾值）
- Close 在 bar 內位置分位（close_pos quartile）作為主要訊號
- Wick 非對稱過濾（上下影線比）

## 均值回歸類（全部飽和）
- Z-score 均值回歸（close vs rolling mean/std）
- Gap fade：隔夜跳空後日內回補操作
- 任何「收盤偏離歷史分位 → 下一棒反向」的結構

## 時間事件類（全部飽和）
- 午休前流動性枯竭偵測作為方向性訊號（可用於禁區，不可作為進場依據）

# 硬性約束
- 年化 trades ∈ [150, 800]（單一補充因子，不是全策略；< 150 樣本太少；> 800 多半 overfit）
- 扣 1.525 點/單摩擦成本後 Sharpe > 0.3
- 與 baseline B (ORB short) 相關性 |ρ| < 0.3
- **與 baseline 合併後**整體 Sharpe 必須 ≥ 1.06（不能讓 baseline 變差；
  原文 1.31 為舊 pipeline 高估值，新 pipeline 下 baseline 自身就只 1.06）
- 時間禁區（多單不可進場，可出場）：
  - 11:00–12:00, 12:30–13:00（亞洲午休，流動性枯竭）
  - 週一/週三 12:00–12:30（結算日洗盤）
  - 週四 08:45–09:00（結算後換倉）
  - 夜盤全部排除，只做日盤

# 過夜持倉鼓勵
- 不強制日內平倉，若 A 軌訊號收盤前仍有效，傾向持倉過夜
- 隔夜 Gap 是重要獲利來源，過夜視為捕捉多日趨勢的機會，不視為風險

# 反例檢測
若新因子核心邏輯（去除參數差異後）與最近 5 個試作「概念骨架」相同，視為無進步，請改方向。

# 方向強制檢查（在生成任何新 hypothesis 前自我確認）
1. 我的因子核心邏輯是 BOCPD 還是 VPIN？如果都不是 → 拒絕生成
2. 我的因子跟 baseline B (ORB short) 的訊號高度同向嗎？如果是 → 拒絕生成
3. 我的因子有沒有重複「已飽和方向」清單裡的概念骨架？如果有 → 拒絕生成
4. 我的因子有沒有用「禁止寫法」的 look-ahead bias pattern？如果有 → 拒絕生成
全部通過才能進入實作。
