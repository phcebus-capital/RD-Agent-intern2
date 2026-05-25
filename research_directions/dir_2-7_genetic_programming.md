# 方向 2-7：Genetic Programming 特徵發現

## 核心

Allen & Karjalainen (1999) *Using Genetic Algorithms to Find Technical Trading Rules*。用 GP 搜尋 {K_Power, K_Pct, OFI, Volume, High-Low} 的非線性組合，找出比 LTR/LCR 更有預測力的特徵。

## 機制

- 個體 = 表達式樹（葉節點 = 原始特徵，內節點 = 算子如 +、−、×、log、rolling_max）
- 適應度 = 樣本內 IC（information coefficient）或 Sharpe
- 操作：crossover（子樹交換）、mutation（節點替換）、tournament selection
- 演化 100-500 代

## 為什麼是中間層

- 比 VPIN / Microprice 是已知公式更開放、可能有 alpha 驚喜
- 比 02 排列熵 / TDA 等需要新理論基礎，GP 是搜尋演算法、工具鏈成熟（gplearn、DEAP）
- **過擬合風險高**，必須搭配 walk-forward + multiple-testing correction

## 對 Pho_Long 的接點

- 把 GP 找出的 top-5 特徵當作「補強訊號」，與現有因子投票
- 進場規則：現有因子至少 N 個正向 → 進場
- 出場規則：原本的退場規則 OR 特定因子全部翻負 → 退場

## 校準難點（這是中間層裡最難的一個）

- **過擬合是天敵**：GP 在訓練集上幾乎一定能找到「完美規則」
- 防護措施：
  1. 訓練 / 驗證 / 測試 三段切割
  2. Deflated Sharpe（López de Prado 2018）做 multiple-testing correction
  3. Probability of Backtest Overfitting (PBO) 檢驗
  4. 限制表達式複雜度（節點數 ≤ 15）防 overfit
- **計算成本**：100 代 × 500 個體 × 幾年資料 ≈ 數小時 GPU

## 文獻

- Allen & Karjalainen (1999) *Using Genetic Algorithms to Find Technical Trading Rules*
- Lo, Mamaysky & Wang (2000) *Foundations of Technical Analysis*
- Bailey, Borwein, López de Prado, Zhu (2014) *Pseudo-Mathematics and Financial Charlatanism*（過擬合警告）
- López de Prado (2018) *Advances in Financial Machine Learning* Ch11-12
