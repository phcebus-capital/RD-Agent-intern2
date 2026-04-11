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

## Notes on instrument codes
Stock codes follow Taiwan market convention: 4-digit numeric strings (e.g. "2330" for TSMC, "2317" for Hon Hai).
ETFs use codes like "0050", "0056".
