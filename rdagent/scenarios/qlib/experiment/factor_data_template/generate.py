"""
Data generator entry point.

Delegates to generate_tw.py (FinLab-based Taiwan stock data).
The output format is identical to the original Qlib cn_data version:
  - daily_pv_all.h5   — full history, all instruments
  - daily_pv_debug.h5 — 2-year debug subset, 100 instruments
Both files use MultiIndex (datetime, instrument) with key="data".

To switch back to the original Qlib cn_data source, replace the import below
with the original code:
    import qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
    from qlib.data import D
    ...
"""

from generate_tw import main

if __name__ == "__main__":
    main()
