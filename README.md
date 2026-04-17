# RD-Agent 台股因子工廠 & 台指期策略研究

RD-Agent 是微軟開源的 LLM 驅動 R&D 自動化框架。本專案以台灣市場為目標，提供兩大應用：

- **台股因子工廠**：以 [Qlib](https://github.com/microsoft/qlib) 作為回測引擎，透過 LLM 自動提出因子假設、撰寫程式、執行回測並學習回饋，循環迭代優化量化因子。
- **台指期（TX）策略研究**：以 1 分鐘 K 棒為數據基礎，自動提出信號因子假設、實作並向量化回測，反覆迭代優化日內交易策略。

> 上游原始碼：[microsoft/RD-Agent](https://github.com/microsoft/RD-Agent)｜[📖 文件](https://rdagent.readthedocs.io/en/latest/index.html)

---

## 一、環境安裝

> 僅支援 Linux。Python 3.10 / 3.11 均可。

```bash
# 建立 Conda 環境
conda create -n rdagent python=3.10
conda activate rdagent

# 安裝套件
pip install rdagent finlab
```

確認 Docker 已安裝，且目前使用者不需要 `sudo` 即可執行 Docker：

```bash
docker run hello-world
```

---

## 二、台股資料準備：轉換為 Qlib 格式

### 2.1 取得 FinLab API Token

前往 [finlab.tw](https://finlab.tw) 註冊帳號並取得 API Token。

- 免費方案：約 3 年歷史資料
- 付費方案：完整歷史（自 2000 年起）

### 2.2 設定 `.env`

複製 [.env.example](.env.example) 為 `.env`，填入以下必要設定：

```bash
# ===== LLM 設定 =====
BACKEND=rdagent.oai.backend.LiteLLMAPIBackend
CHAT_MODEL=gpt-4.1
OPENAI_API_KEY=<your_openai_api_key>
OPENAI_API_BASE=https://api.openai.com/v1

# Embedding 模型
EMBEDDING_MODEL=text-embedding-3-small

# ===== 台股資料 =====
FINLAB_API_TOKEN=<your_finlab_api_token>

# ===== 環境 =====
CONDA_DEFAULT_ENV=rdagent
```

### 2.3 執行資料轉換

```bash
bash generate_tw.sh
```

腳本執行流程：

| 步驟 | 說明 |
|------|------|
| 1 | 從 `.env` 載入環境變數 |
| 2 | 呼叫 [generate_tw.py](rdagent/scenarios/qlib/experiment/factor_data_template/generate_tw.py)，透過 FinLab API 下載台股資料 |
| 3 | 產出 `daily_pv_all.h5`（全量）與 `daily_pv_debug.h5`（調試用） |
| 4 | 輸出 Qlib binary 格式到 `~/.qlib/qlib_data/tw_data/` |
| 5 | 複製 HDF5 到 `git_ignore_folder/factor_implementation_source_data/` |

### 2.4 產出檔案

| 檔案 / 目錄 | 說明 |
|------------|------|
| `daily_pv_all.h5` | 全量台股資料（2010-01-01 起，所有股票） |
| `daily_pv_debug.h5` | 調試資料（2020–2021，流動性前 100 檔） |
| `~/.qlib/qlib_data/tw_data/` | Qlib binary provider（回測使用） |
| `git_ignore_folder/factor_implementation_source_data/` | RD-Agent 因子實作讀取路徑 |

讀取 HDF5 方式：

```python
import pandas as pd
df = pd.read_hdf("daily_pv_all.h5", key="data")
# Index: MultiIndex(datetime, instrument)
```

### 2.5 資料欄位說明

**每日行情**

| 欄位 | 說明 |
|------|------|
| `$open` | 開盤價（TWD） |
| `$close` | 收盤價（TWD） |
| `$high` | 最高價（TWD） |
| `$low` | 最低價（TWD） |
| `$volume` | 成交股數（股） |
| `$factor` | 復權因子 = adj_close / close |
| `$market_cap` | 市值（TWD），自 2013-04-19，約 20% NaN |
| `$foreign_net` | 外資買賣超股數（正=買超，負=賣超） |
| `$trust_net` | 投信買賣超股數 |
| `$dealer_net` | 自營商（避險）買賣超股數 |

**月營收**（每月公告後自動 forward-fill 至每個交易日）

| 欄位 | 說明 |
|------|------|
| `$mr_cur` | 當月營收（TWD） |
| `$mr_prev` | 上月營收 |
| `$mr_yoy` | 去年同月營收 |
| `$mr_mom_pct` | 月增率（%） |
| `$mr_yoy_pct` | 年增率（%） |
| `$mr_cum` | 年累計營收 |
| `$mr_cum_yoy` | 去年年累計營收 |
| `$mr_cum_pct` | 年累計年增率（%） |

> 股票代碼為 4 碼數字字串，例如 `"2330"`（台積電）、`"0050"`（台灣 50 ETF）。  
> `0050` 會自動建立別名 `TW0050` 供回測使用。

---

## 三、RD-Agent 設定與調教

### 3.1 訓練 / 驗證 / 測試區間

透過 `QLIB_FACTOR_*` 環境變數設定（可寫入 `.env`）：

```bash
# 訓練區間（預設 2010-01-01 ~ 2018-12-31）
QLIB_FACTOR_TRAIN_START=2010-01-01
QLIB_FACTOR_TRAIN_END=2020-12-31

# 驗證區間（預設 2019-01-01 ~ 2020-12-31）
QLIB_FACTOR_VALID_START=2021-01-01
QLIB_FACTOR_VALID_END=2022-12-31

# 測試 / 回測區間（預設 2021-01-01 起）
QLIB_FACTOR_TEST_START=2023-01-01
# QLIB_FACTOR_TEST_END=2024-12-31   # 不填則使用資料最末日

# 每輪 CoSTEER 演化次數（預設 10）
QLIB_FACTOR_EVOLVING_N=10
```

> **注意**：FinLab 免費方案僅約 3 年資料，請依實際資料範圍縮短訓練區間，或升級付費方案。

### 3.2 因子實作參數

```bash
# 資料目錄（generate_tw.sh 執行後已自動填充）
FACTOR_CoSTEER_DATA_FOLDER=git_ignore_folder/factor_implementation_source_data
FACTOR_CoSTEER_DATA_FOLDER_DEBUG=git_ignore_folder/factor_implementation_source_data_debug

# 每次因子執行 timeout（秒）
FACTOR_CoSTEER_FILE_BASED_EXECUTION_TIMEOUT=3600

# 因子候選選取策略：random / topk
FACTOR_CoSTEER_SELECT_METHOD=random
```

---

## 四、啟動台股因子工廠

### 4.1 確認設定

```bash
rdagent health_check
```

### 4.2 啟動

```bash
# 方法 A：使用 run.sh（推薦，自動載入 .env）
bash run.sh

# 方法 B：手動執行
export $(grep -v '^#' .env | xargs)
rdagent fin_factor --loop-n 30
```

`--loop-n 30`：執行 30 輪迭代，每輪包含：假設提出 → 因子撰寫 → 回測執行 → 回饋學習。

### 4.3 監控執行結果

```bash
# Streamlit UI
rdagent ui --port 19899 --log-dir log/

# Web UI（Flask）
rdagent server_ui --port 19899
# 開啟瀏覽器：http://127.0.0.1:19899
```

---

## 五、台指期（TX）1 分鐘策略研究

本模組以台指期近月合約的 1 分鐘 K 棒為數據源，透過 RD-Agent 循環自動研究日內信號因子。

### 5.1 資料來源

原始 tick 資料從 [FinMind](https://finmindtrade.com/) 下載，清洗並聚合為 1 分鐘 K 棒後，存放於 `tw-futures-data/data_clean/kbar_1min_clean/`。詳細的資料下載與清洗流程請參閱 [tw-futures-data/README.md](tw-futures-data/README.md)。

| 欄位 | 說明 |
|------|------|
| `open` | 開盤價 |
| `high` | 最高價 |
| `low` | 最低價 |
| `close` | 收盤價 |
| `volume` | 成交量（口） |
| `session` | 盤別：`day`（日盤 08:45–13:45）/ `night`（夜盤 15:00–隔日 05:00） |

### 5.2 準備資料（執行一次）

```bash
# 將每日 CSV 合併為 Parquet 供 RD-Agent 使用
python rdagent/scenarios/futures/prepare_data.py
```

預設輸出路徑：

| 路徑 | 說明 |
|------|------|
| `git_ignore_folder/futures_source_data/data.parquet` | 全量資料集 |
| `git_ignore_folder/futures_source_data_debug/data.parquet` | 調試資料集（前 60 個交易日） |

可選參數：

```bash
python rdagent/scenarios/futures/prepare_data.py \
  --kbar_dir tw-futures-data/data_clean/kbar_1min_clean \
  --out_dir   git_ignore_folder/futures_source_data \
  --debug_dir git_ignore_folder/futures_source_data_debug \
  --debug_days 60
```

### 5.3 設定環境變數（寫入 `.env`）

```bash
# 資料目錄
FUTURES_CoSTEER_data_folder=git_ignore_folder/futures_source_data
FUTURES_CoSTEER_data_folder_debug=git_ignore_folder/futures_source_data_debug

# 訓練 / 驗證 / 測試區間（以下為預設值，可依需求修改）
FUTURES_FACTOR_train_start=2020-01-02
FUTURES_FACTOR_train_end=2022-12-31
FUTURES_FACTOR_valid_start=2023-01-01
FUTURES_FACTOR_valid_end=2024-06-30
FUTURES_FACTOR_test_start=2024-07-01
# FUTURES_FACTOR_test_end=2026-04-15   # 不填則使用資料最末日

# 每輪 CoSTEER 演化次數（預設 10）
# FUTURES_FACTOR_evolving_n=10
```

### 5.4 啟動策略研究循環

```bash
# 全新開始
python rdagent/app/futures_rd_loop/factor.py

# 續跑上次的 session
python rdagent/app/futures_rd_loop/factor.py --path LOG_PATH/__session__/1/0_propose

# 限制執行輪數或時長
python rdagent/app/futures_rd_loop/factor.py --loop-n 20
python rdagent/app/futures_rd_loop/factor.py --all-duration 2h
```

每輪流程：**假設提出 → 因子程式撰寫 → 向量化回測 → Sharpe / 最大回撤回饋 → 下一輪假設**。

### 5.5 LLM 生成的因子格式

每個因子為一個獨立的 `factor.py`，輸入為 `data.parquet`，輸出為 `result.h5`：

```python
# 輸入
import pandas as pd
df = pd.read_parquet("data.parquet")
# Index: DatetimeIndex（1 分鐘），欄位：open / high / low / close / volume / session

# 輸出（key="signal"）：正值 = 做多信號，負值 = 做空信號，~0 = 空倉
signal: pd.Series  # 與 df 相同索引
signal.to_hdf("result.h5", key="signal")
```

---

## 完整啟動流程

### 台股因子工廠

```
1. conda activate rdagent
2. 設定 .env（FINLAB_API_TOKEN、OPENAI_API_KEY...）
3. bash generate_tw.sh       ← 下載台股資料並轉換 Qlib 格式
4. rdagent health_check      ← 確認 LLM 及 Docker 正常
5. bash run.sh               ← 啟動因子工廠（30 輪迭代）
6. rdagent ui --port 19899 --log-dir log/   ← 監控結果
```

### 台指期策略研究

```
1. conda activate rdagent
2. 設定 .env（OPENAI_API_KEY、FUTURES_CoSTEER_data_folder...）
3. python rdagent/scenarios/futures/prepare_data.py   ← 準備 Parquet 資料（一次）
4. rdagent health_check                               ← 確認 LLM 正常
5. python rdagent/app/futures_rd_loop/factor.py       ← 啟動策略研究循環
6. rdagent ui --port 19899 --log-dir log/             ← 監控結果
```
