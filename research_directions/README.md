# research_directions/

存放台指期（TX）1-min 策略研究的「研究方向」種子檔。每個 `.md` 是一份完整的研究方向描述，
會被 `run.sh` 當成 round-0 的策略種子（`--strategy_file`）餵給 RD-Loop 的假設生成器。

## 用法

```bash
# 列出所有可用方向
bash run.sh

# 用指定方向啟動 RD-Loop（檔名可省略 .md）
bash run.sh bocpd_vpin_complement
bash run.sh microstructure_exploration

# 覆寫迴圈數（預設 300）
LOOP_N=50 bash run.sh bocpd_vpin_complement
```

## 撰寫慣例

- 檔名用 kebab/snake-case 的 slug，例如 `bocpd_vpin_complement.md`。
- 第一行建議是 `#` 開頭的標題，`bash run.sh`（不帶參數）會把它當成方向的一行說明。
- 內容直接寫給 LLM 看：目標、強制方向、已飽和（請避開）、硬性約束、自我檢查清單。
- 這份文字會被原樣注入 `FuturesFactorHypothesisGenFromStrategy` 當 round-0 種子，
  之後 LLM 會依 trace 歷史自行迭代——種子是「軟偏好」，不是逐字實作規格。

## 現有方向

| 檔案 | 重點 |
|------|------|
| `bocpd_vpin_complement.md` | 只接受 BOCPD / VPIN 概念，補足現有 baseline (B-override + A) 的缺口 |
| `microstructure_exploration.md` | 較寬鬆的微結構 / 事件 / 路徑相依探索，trades ∈ [400,1500] |
