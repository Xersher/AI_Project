"""
evaluate.py: alphas of monthly strategy returns against the seven benchmark models.

Usage as a script (from the project folder):
    python code/evaluate.py output/my_strategy.csv [--long-only] [--nw LAGS]
    my_strategy.csv: a column YYYYMM plus one or more return columns (decimals), e.g. ls_gross, ls_net.
    Factors are read from data/processed/factors_monthly.parquet (paths.py); another file can be
    given as a second argument.
Usage as a module (from a script in code/):
    from evaluate import alpha_table, load_factors
    table = alpha_table(returns_series, load_factors())

Conventions (project decisions, context_ai_project.md sections 4-5):
  - Long-short (zero-investment) returns are regressed as they are; long-only returns are first turned
    into excess returns over RF (--long-only).
  - Headline: FF6 alpha with OLS t-statistic; the HXZ multiple-testing hurdle is |t| >= 2.78.
    Newey-West t-statistics are optional robustness (--nw LAGS).
  - Each model uses every month where the strategy and all of that model's factors are available,
    so N and the sample dates are reported per model.
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

import paths

MODELS = {
    "CAPM": ["mktrf"],
    "FF3": ["mktrf", "smb", "hml"],
    "FF3+Mom": ["mktrf", "smb", "hml", "umd"],
    "FF5": ["mktrf5", "smb5", "hml5", "rmw", "cma"],
    "FF6": ["mktrf5", "smb5", "hml5", "rmw", "cma", "umd"],
    "q": ["q_mkt", "q_me", "q_ia", "q_roe"],
    "q5": ["q_mkt", "q_me", "q_ia", "q_roe", "q_eg"],
}
HURDLE = 2.78 # Change this hurdle to test if the strategy works (1.96 for 5% significance, 2.78 for HXZ multiple-testing)


def load_factors(path: str | None = None) -> pd.DataFrame:
    """Monthly factors indexed by YYYYMM; default file from paths.py."""
    path = str(path or paths.FACTORS)
    f = duckdb.read_parquet(path).df() if path.endswith(".parquet") else pd.read_csv(path)
    return f.set_index("YYYYMM").sort_index()


def ols(y: np.ndarray, x: np.ndarray, nw_lags: int | None = None) -> dict:
    """OLS with an intercept. Returns coefficients, OLS t-stats, optional Newey-West t-stats, R2."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    e = y - X @ b
    k = X.shape[1]
    se = np.sqrt(np.diag(xtx_inv) * (e @ e) / (n - k))
    out = {"b": b, "t": b / se, "r2": 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())), "n": n,
           "resid_sd": np.sqrt((e @ e) / (n - k))}
    if nw_lags:
        xe = X * e[:, None]
        s = xe.T @ xe
        for lag in range(1, nw_lags + 1):
            w = 1 - lag / (nw_lags + 1)  # Bartlett kernel
            g = xe[lag:].T @ xe[:-lag]
            s += w * (g + g.T)
        cov = xtx_inv @ s @ xtx_inv * n / (n - k)
        out["t_nw"] = b / np.sqrt(np.diag(cov))
    return out


def alpha_table(ret: pd.Series, factors: pd.DataFrame, long_only: bool = False,
                nw_lags: int | None = None) -> pd.DataFrame:
    """One row per model: monthly alpha (%), t-stats, R2, N, sample, factor loadings."""
    y_all = ret.astype(float)
    if long_only:
        y_all = y_all - factors["rf"].reindex(y_all.index)
    rows = []
    for name, cols in MODELS.items():
        d = pd.concat([y_all.rename("y"), factors[cols]], axis=1, join="inner").dropna()
        if len(d) < 24:
            continue
        r = ols(d["y"].to_numpy(), d[cols].to_numpy(), nw_lags)
        row = {"model": name, "alpha_pct": 100 * r["b"][0], "t_ols": r["t"][0]}
        if nw_lags:
            row["t_nw"] = r["t_nw"][0]
        row.update({"r2": r["r2"], "n": r["n"], "first": int(d.index.min()), "last": int(d.index.max()),
                    "ann_appraisal": r["b"][0] / r["resid_sd"] * np.sqrt(12),
                    "passes_2.78": abs(r["t"][0]) >= HURDLE})
        row.update({f"b_{c}": v for c, v in zip(cols, r["b"][1:])})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strategy_csv")
    ap.add_argument("factors", nargs="?", default=None, help="default: data/processed/factors_monthly.parquet")
    ap.add_argument("--long-only", action="store_true", help="subtract RF before regressing")
    ap.add_argument("--nw", type=int, default=None, help="Newey-West lags for robustness t-stats")
    a = ap.parse_args()
    s = pd.read_csv(a.strategy_csv).set_index("YYYYMM").sort_index()
    f = load_factors(a.factors)
    pd.set_option("display.width", 200)
    for col in s.columns:
        print(f"\n=== {col}: mean {100 * s[col].mean():.3f}% per month, "
              f"t(mean) {s[col].mean() / s[col].std() * np.sqrt(s[col].count()):.2f}, "
              f"{s[col].count()} months ===")
        tab = alpha_table(s[col], f, a.long_only, a.nw)
        main_cols = [c for c in tab.columns if not c.startswith("b_")]
        print(tab[main_cols].round(3).to_string())


if __name__ == "__main__":
    main()