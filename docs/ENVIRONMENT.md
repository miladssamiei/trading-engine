# Publication verification environment

The source release was checked locally on Windows with Python **3.12.13**.
The project environment already contained these packages:

| Package | Installed version |
| --- | --- |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| Numba | 0.61.2 |
| SciPy | 1.17.1 |
| matplotlib | 3.11.0 |
| scikit-learn | 1.9.0 |
| statsmodels | 0.14.6 |
| XGBoost | 3.4.1 |
| LightGBM | 4.6.0 |
| MetaTrader5 | 5.0.6090 |
| pytest | 9.0.3 |

This is a record of the checked environment, not a claim that every historical
study originally used these versions. Project dependency ranges are in
`pyproject.toml`; optional packages are separated from the offline core.

Local publication verification included 49 passing offline tests, the synthetic
execution example, source syntax checks, local documentation links, quoted MQL5
include resolution and the publication file/hash audit. No fresh broker-data
download, full historical rerun, live order or MetaEditor compilation was part
of this packaging task. Configured GitHub CI platform/version combinations need
their own actual CI runs after publication.
