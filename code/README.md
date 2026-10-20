# code/: how the scripts fit together

Run every command from the project root (`applied_investments/`). Data never lives in this folder.

## Setup (once)
```
pip install -r code/requirements.txt
```
Python 3.9 or newer.

## 1. Data preparation: run once, in this order

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `data_audit.py` | the single CSV in `data/raw/crsp/` | `data/processed/crsp_monthly.parquet` + audit report (`output/logs/data_audit.txt`) |
| 2 | `build_panel.py` | `data/processed/crsp_monthly.parquet` | `data/processed/crsp_panel.parquet` (the clean panel all later code uses) |
| 3 | `validate_vs_french.py` | `data/processed/crsp_panel.parquet` (+ French size and momentum files, downloaded automatically) | correlation tables (save the printout in `output/logs/`) |
| 4 | `get_factors.py` | the 4 files in `data/raw/factors/` | `data/processed/factors_monthly.parquet/.csv`, `factors_vintage.txt` (**already done 16 Oct**) |

```
python code/data_audit.py
python code/build_panel.py data/processed/crsp_monthly.parquet
python code/validate_vs_french.py data/processed/crsp_panel.parquet
python code/get_factors.py data/processed data/raw/factors
```

## 2. Shared tools: imported by the strategy scripts
| Module | What it does |
|---|---|
| `evaluate.py` | Alphas against CAPM, FF3, FF3+Mom, FF5, FF6, q, q5: OLS t (headline), optional Newey–West, R², N, sample per model, 2.78 flag. Also usable from the command line on a CSV of returns. |
| *to come:* sorting engine, transaction costs, Fama–MacBeth | |

## 3. Strategy scripts (the team's)
One script per idea, e.g. `strategy_<name>.py`. Each reads `crsp_panel.parquet` and `factors_monthly.parquet`, writes its portfolio returns to `output/`, then calls `evaluate.py`. Log every specification you run in §12 of `context_ai_project.md`.

## Rules
- Never edit anything in `data/raw/`.
- Raw data and generated data never go into git (`data/` is in `.gitignore`).
- Every number in the paper must come from a script in this folder.
