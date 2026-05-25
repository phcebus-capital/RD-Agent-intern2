# 方向 7：Regime-Dependent Adaptive Threshold

**核心想法**：把 LTR / LCR 的 `65` 從硬閾值改成「動態分位數」。

```python
ltr_entry = rolling_quantile(LTR_history, 0.85, lookback=60d)
lcr_entry = rolling_quantile(LCR_history, 0.85, lookback=60d)
```

## 為什麼有效

- 多頭主升段 LTR 普遍高 → `65` 太低 → 訊號被稀釋
- 震盪市 LTR 普遍低 → `65` 太高 → 訊號餓死
- 動態分位數讓「相對強度」恆定

## 文獻

- **Asness, Israelov & Liew (2011)** *International Diversification Works (Eventually)*
  — 提到動態 z-score 比 fixed threshold 在不同 regime 下穩健
- **Pedersen (2015)** *Efficient Asymmetric Trading* 第 5 章

## 注意

不是參數掃描！是把「常數」換成「rolling statistic」，是**訊號層的結構升級**。
