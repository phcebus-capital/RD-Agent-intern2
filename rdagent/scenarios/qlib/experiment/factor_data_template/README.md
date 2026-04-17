# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | Adjusted daily price, volume, and market cap data (Taiwan stock market). |


# For different data, We have some basic knowledge for them

## Daily price and volume data
$open: open price of the stock on that day (TWD).
$close: close price of the stock on that day (TWD).
$high: high price of the stock on that day (TWD).
$low: low price of the stock on that day (TWD).
$volume: trading volume of the stock on that day (unit: 股).
$factor: price adjustment factor for dividends and splits (adjusted_close / close). Use this to convert raw prices to cumulative adjusted prices.
$market_cap: market capitalization of the stock on that day (TWD). Available from 2013-04-19 onward. May be NaN for newly listed or suspended stocks (~20% NaN rate overall).
$foreign_net: net buy/sell volume by foreign institutional investors on that day (unit: shares, 股). Positive = net buy, negative = net sell. May be NaN for some days.
$trust_net: net buy/sell volume by investment trust funds (投信) on that day (unit: shares, 股). Positive = net buy, negative = net sell. May be NaN for some days.
$dealer_net: net buy/sell volume by securities dealers (自營商, hedging only) on that day (unit: shares, 股). Positive = net buy, negative = net sell. May be NaN for some days.

## Monthly revenue data (月營收)
Monthly revenue is disclosed by listed companies by the 10th of the following month (or the next trading day when the 10th falls on a holiday). All $mr_* columns are forward-filled from the announcement date, so each trading day carries the most recently disclosed figure. Data available from 2005-02-10.

$mr_cur: current-month revenue in TWD (當月營收). Updates once per month on the announcement date.
$mr_prev: previous-month revenue in TWD (上月營收).
$mr_yoy: same-month revenue from one year ago in TWD (去年當月營收).
$mr_mom_pct: month-over-month revenue change in % (上月比較增減%).
$mr_yoy_pct: year-over-year same-month revenue change in % (去年同月增減%).
$mr_cum: year-to-date cumulative revenue in TWD (當月累計營收).
$mr_cum_yoy: year-to-date cumulative revenue from last year in TWD (去年累計營收).
$mr_cum_pct: year-to-date cumulative revenue YoY change in % (前期比較增減%).

## Notes on instrument codes
Stock codes follow Taiwan market convention: 4-digit numeric strings (e.g. "2330" for TSMC, "2317" for Hon Hai).
ETFs use codes like "0050", "0056".
