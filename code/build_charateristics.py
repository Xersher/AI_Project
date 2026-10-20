"""
build_characteristics.py: the 18 CRSP-only stock characteristics used by strategy_composite.py.

Usage (from the project folder):   python code/build_characteristics.py
  Reads   data/processed/crsp_panel.parquet       (build_panel.py)
          data/processed/factors_monthly.parquet  (get_factors.py; FF3 factors and RF)
  Writes  data/processed/characteristics.parquet  one row per PERMNO and formation month tau
  Log     output/logs/build_characteristics.txt

Timing rule (no look-ahead): every characteristic dated tau uses only data up to the end of month tau.
The strategy forms portfolios at the end of tau and earns the returns of month t = tau + 1.
Windows below are written relative to the holding month t, as in the literature.

Candidates at tau: rows of the panel with a common-stock row at tau (build_panel's `common`, which already
requires NYSE/AMEX/Nasdaq and active trading), a positive market cap and a positive price.

The 18 characteristics (all windows 36 months or shorter, so the first full set exists in Dec 1970):
  size        log market cap at tau
  price       log price at tau
  rev1        return of month t-1 (= tau)
  mom12_2     cumulative log return t-12..t-2              (>= 8 of 11 months)
  mom12_7     cumulative log return t-12..t-7              (>= 5 of 6)
  ltrev36_13  cumulative log return t-36..t-13             (>= 18 of 24)
  season      mean return in months t-12, t-24, t-36       (>= 2 of 3; same calendar month as t)
  ind_mom6    value-weighted 2-digit-SIC industry return over t-6..t-1 (>= 5 of 6; >= 5 stocks)
  ind_rel1    own return minus own industry's value-weighted return, month t-1
  resmom      mean FF3 residual over t-12..t-2 divided by its standard deviation; FF3 betas from the
              36 months t-36..t-1 (>= 24 months; >= 8 residuals)  [Blitz, Huij & Martens 2011 style]
  nissue12    12-month net issuance: log ME growth minus cumulative log price return (all 12 months)
  cissue36    36-month composite issuance: log ME growth minus cumulative log total return (all 36)
  dy12        dividend return over t-12..t-1: sum of (MthRet - MthRetx)      (>= 10 of 12)
  beta36      CAPM beta on Mkt-RF over t-36..t-1           (>= 24 of 36)
  ivol36      standard deviation of FF3 residuals, t-36..t-1 (>= 24 of 36)
  turn12      mean monthly share turnover t-12..t-1         (>= 9 of 12); Nasdaq volume adjusted, see below
  maxret12    maximum monthly return t-12..t-1              (>= 9 of 12)
  hi12        log of price index at t-1 relative to its 12-month maximum (dividend-free index from MthRetx;
              split-proof)                                  (>= 9 of 12 MthRetx)

Missing data stay missing (NaN). The strategy script decides how to treat them (disclosed there).

NASDAQ volume: Nasdaq volume double-counts dealer trades before 2004. Following the adjustment commonly
attributed to Gao & Ritter (2010, JFE) -- VERIFY against the paper before citing -- Nasdaq volume is divided
by 2.0 before Feb 2001, 1.8 for Feb-Dec 2001, 1.6 for 2002-2003, and 1.0 from 2004. Set
ADJUST_NASDAQ_VOLUME = False to switch it off.

Any change to a window, minimum count or definition is a new specification: log it in section 12 of
context_ai_project.md before running.
"""
from __future__ import annotations

import sys
import time

import duckdb
import numpy as np
import pandas as pd

import paths

ADJUST_NASDAQ_VOLUME = True
MIN_INDUSTRY_STOCKS = 5
OUT_NAME = "characteristics.parquet"
CHUNK_PERMNOS = 3000          # stocks processed at a time (lower it if memory is tight)
CHARS = ["size", "price", "rev1", "mom12_2", "mom12_7", "ltrev36_13", "season", "ind_mom6", "ind_rel1",
         "resmom", "nissue12", "cissue36", "dy12", "beta36", "ivol36", "turn12", "maxret12", "hi12"]


# ----------------------------------------------------------------------------------------------------------
# Helpers on a dense panel: rows sorted by PERMNO then month, one row for EVERY month between a stock's
# first and last panel month (gaps filled with NaN), so "k rows back" always means "k months back".
# ----------------------------------------------------------------------------------------------------------
class Dense:
    def __init__(self, permno: np.ndarray, mi: np.ndarray):
        self.permno, self.mi = permno, mi
        self.n = len(mi)
        new_group = np.r_[True, permno[1:] != permno[:-1]]
        self.start = np.maximum.accumulate(np.where(new_group, np.arange(self.n), 0))  # first row of group
        self.pos = np.arange(self.n) - self.start                                      # months since start

    def lag(self, x: np.ndarray, k: int) -> np.ndarray:
        """Value k months earlier (NaN if before the stock's first month)."""
        out = np.full(self.n, np.nan)
        if k == 0:
            return x.astype(float).copy()
        ok = self.pos >= k
        out[ok] = x[np.flatnonzero(ok) - k]
        return out

    def _cum(self, x0: np.ndarray) -> np.ndarray:
        cs = np.cumsum(x0)
        before_start = np.where(self.start > 0, cs[np.maximum(self.start - 1, 0)], 0.0)
        return cs, before_start

    def rsum(self, x: np.ndarray, w: int) -> np.ndarray:
        """Sum of the non-missing values over the w months ending at each row (inclusive)."""
        cs, base = self._cum(np.nan_to_num(x, nan=0.0))
        out = cs - base
        full = self.pos >= w
        idx = np.flatnonzero(full)
        out[idx] = cs[idx] - cs[idx - w]
        return out

    def rcount(self, x: np.ndarray, w: int) -> np.ndarray:
        return self.rsum((~np.isnan(x)).astype(float), w)

    def gcumsum(self, x: np.ndarray) -> np.ndarray:
        cs, base = self._cum(np.nan_to_num(x, nan=0.0))
        return cs - base

    def rmax(self, x: np.ndarray, w: int, min_obs: int) -> np.ndarray:
        s = pd.Series(x).groupby(self.permno).rolling(w, min_periods=min_obs).max()
        return s.reset_index(level=0, drop=True).sort_index().to_numpy()


def rolling_ols(d: Dense, y: np.ndarray, X: np.ndarray, w: int, min_obs: int):
    """Rolling OLS of y on X (X includes the constant) over the w months ending at each row.
    Returns betas (n x k), number of observations, and the rolling sums needed for residual moments."""
    n, k = X.shape
    m = (~np.isnan(y)) & ~np.isnan(X).any(axis=1)
    ym = np.where(m, y, 0.0)
    Xm = np.where(m[:, None], X, 0.0)
    nobs = d.rsum(m.astype(float), w)
    sxx = np.empty((n, k, k))
    for a in range(k):
        for b in range(a, k):
            sxx[:, a, b] = d.rsum(Xm[:, a] * Xm[:, b], w)
            sxx[:, b, a] = sxx[:, a, b]
    sxy = np.column_stack([d.rsum(Xm[:, a] * ym, w) for a in range(k)])
    syy = d.rsum(ym * ym, w)
    beta = np.full((n, k), np.nan)
    ok = nobs >= min_obs
    if ok.any():
        det = np.linalg.det(sxx[ok])
        good = np.flatnonzero(ok)[np.abs(det) > 1e-12]
        beta[good] = np.linalg.solve(sxx[good], sxy[good][..., None])[..., 0]
    return beta, nobs, sxx, sxy, syy


def industry_tables(p: pd.DataFrame):
    """Value-weighted 2-digit-SIC industry returns per month (weights: the stock's previous-month market cap,
    only if that month is in the panel) and their 6-month cumulative log return. Needs the whole panel."""
    prev_mi = p.groupby("PERMNO").mi.shift(1)
    me_lag = p.groupby("PERMNO").me.shift(1).where(prev_mi == p.mi - 1)
    sic2 = np.where(p.sic > 0, np.floor(p.sic / 100), np.nan)
    ok = p.common.fillna(False).astype(bool) & p.ret.notna() & (me_lag > 0) & ~np.isnan(sic2)
    ind = pd.DataFrame({"sic2": sic2[ok], "mi": p.mi[ok], "w": me_lag[ok], "wr": (me_lag * p.ret)[ok]})
    ind = ind.groupby(["sic2", "mi"]).agg(w=("w", "sum"), wr=("wr", "sum"), n=("w", "size"))
    ind["r"] = np.where(ind.n >= MIN_INDUSTRY_STOCKS, ind.wr / ind.w, np.nan)
    wide = ind.r.unstack("sic2").reindex(range(int(p.mi.min()), int(p.mi.max()) + 1))
    mom6 = np.log(np.maximum(1 + wide, 1e-6)).rolling(6, min_periods=5).sum()   # tau-5..tau = t-6..t-1
    idx = pd.MultiIndex.from_product([wide.index, wide.columns])
    return pd.Series(wide.to_numpy().ravel(), index=idx), pd.Series(mom6.to_numpy().ravel(), index=idx)


def chunk_characteristics(p: pd.DataFrame, f: pd.DataFrame, r_ind: pd.Series, m6: pd.Series):
    """All 18 characteristics for the stocks in p (a set of whole PERMNO histories)."""
    # ---- dense grid --------------------------------------------------------------------------------------
    g = p.groupby("PERMNO").mi.agg(["min", "max"])
    lens = (g["max"] - g["min"] + 1).to_numpy()
    permno = np.repeat(g.index.to_numpy(), lens)
    offs = np.r_[0, np.cumsum(lens)[:-1]]
    mi = np.repeat(g["min"].to_numpy(), lens) + (np.arange(lens.sum()) - np.repeat(offs, lens))
    d = Dense(permno, mi)
    loc = (pd.Series(offs, index=g.index).reindex(p.PERMNO).to_numpy()
           + p.mi.to_numpy() - g["min"].reindex(p.PERMNO).to_numpy())

    def col(name):
        out = np.full(d.n, np.nan)
        out[loc] = p[name].to_numpy(dtype=float, na_value=np.nan)
        return out

    ret, ret_crsp, retx, me, prc, turn, sic = (col(c) for c in
                                               ("ret", "ret_crsp", "retx", "me", "prc", "turnover", "sic"))
    exch = np.full(d.n, None, dtype=object)
    exch[loc] = p["exch"].to_numpy(dtype=object)
    in_panel = np.zeros(d.n, bool)
    in_panel[loc] = True
    common = np.zeros(d.n, bool)
    common[loc] = p["common"].fillna(False).astype(bool).to_numpy()
    yyyymm = (mi // 12) * 100 + mi % 12 + 1
    gaps = d.n - len(p)

    fm = f.set_index("mi").reindex(mi)
    rf, mkt, smb, hml = (fm[c].to_numpy(float) for c in ("rf", "mktrf", "smb", "hml"))
    exret = ret - rf
    if ADJUST_NASDAQ_VOLUME:
        div = np.select([yyyymm < 200102, yyyymm < 200201, yyyymm < 200401], [2.0, 1.8, 1.6], 1.0)
        turn = np.where(exch == "Q", turn / div, turn)
    logret = np.log(np.maximum(1 + ret, 1e-6))
    logretx = np.log(np.maximum(1 + retx, 1e-6))
    ch = {}

    def windowed(x, lag, w, min_obs):
        xl = d.lag(x, lag)
        s, c = d.rsum(xl, w), d.rcount(xl, w)
        return np.where(c >= min_obs, s, np.nan)

    # ---- size, price, past returns -----------------------------------------------------------------------
    ch["size"] = np.where(me > 0, np.log(np.where(me > 0, me, 1)), np.nan)
    ch["price"] = np.where(prc > 0, np.log(np.where(prc > 0, prc, 1)), np.nan)
    ch["rev1"] = ret.copy()
    ch["mom12_2"] = windowed(logret, 1, 11, 8)        # t-12..t-2  = tau-11..tau-1
    ch["mom12_7"] = windowed(logret, 6, 6, 5)         # t-12..t-7  = tau-11..tau-6
    ch["ltrev36_13"] = windowed(logret, 12, 24, 18)   # t-36..t-13 = tau-35..tau-12
    lags = np.column_stack([d.lag(ret, k) for k in (11, 23, 35)])  # t-12, t-24, t-36
    nlag = (~np.isnan(lags)).sum(axis=1)
    with np.errstate(invalid="ignore"):
        ch["season"] = np.where(nlag >= 2, np.nansum(lags, axis=1) / np.maximum(nlag, 1), np.nan)

    # ---- industries (2-digit SIC at tau) -----------------------------------------------------------------
    sic2 = np.where(sic > 0, np.floor(sic / 100), np.nan)
    key = pd.MultiIndex.from_arrays([mi, sic2])
    ch["ind_mom6"] = m6.reindex(key).to_numpy()
    ch["ind_rel1"] = ret - r_ind.reindex(key).to_numpy()

    # ---- regressions: CAPM beta, FF3 residual volatility, residual momentum ------------------------------
    # Windows t-36..t-1 = tau-35..tau: the 36 months ending at row tau, no lag.
    one = np.ones(d.n)
    b_capm, *_ = rolling_ols(d, exret, np.column_stack([one, mkt]), 36, 24)
    ch["beta36"] = b_capm[:, 1]
    X3 = np.column_stack([one, mkt, smb, hml])
    b3, n3, sxx, sxy, syy = rolling_ols(d, exret, X3, 36, 24)
    del sxx
    ssr = syy - np.einsum("ij,ij->i", np.nan_to_num(b3), sxy)
    with np.errstate(invalid="ignore", divide="ignore"):
        ch["ivol36"] = np.where(n3 >= 24, np.sqrt(np.maximum(ssr, 0) / np.maximum(n3 - 4, 1)), np.nan)
    # Residual momentum: residuals of t-12..t-2 (= tau-11..tau-1) with the betas of t-36..t-1.
    # Lagging y and X by one month, an 11-month window ending at tau covers exactly tau-11..tau-1.
    y11, X11 = d.lag(exret, 1), d_lag_X(d, X3, 1)
    m11 = (~np.isnan(y11)) & ~np.isnan(X11).any(axis=1)
    y11m = np.where(m11, y11, 0.0)
    X11m = np.where(m11[:, None], X11, 0.0)
    n11 = d.rsum(m11.astype(float), 11)
    bb = np.nan_to_num(b3)
    sy11, syy11 = d.rsum(y11m, 11), d.rsum(y11m * y11m, 11)
    sx = np.column_stack([d.rsum(X11m[:, a], 11) for a in range(4)])
    sxy11 = np.column_stack([d.rsum(X11m[:, a] * y11m, 11) for a in range(4)])
    quad = np.zeros(d.n)                              # b' (sum x x') b without storing the 4x4 matrices
    for a in range(4):
        for b in range(4):
            quad += bb[:, a] * bb[:, b] * d.rsum(X11m[:, a] * X11m[:, b], 11)
    res_sum = sy11 - np.einsum("ij,ij->i", bb, sx)
    res_ss = syy11 - 2 * np.einsum("ij,ij->i", bb, sxy11) + quad
    with np.errstate(invalid="ignore", divide="ignore"):
        res_var = (res_ss - res_sum ** 2 / n11) / (n11 - 1)
        ch["resmom"] = np.where((n11 >= 8) & (n3 >= 24) & (res_var > 0),
                                (res_sum / n11) / np.sqrt(res_var), np.nan)

    # ---- issuance, payout, trading -----------------------------------------------------------------------
    lme = np.where(me > 0, np.log(np.where(me > 0, me, 1)), np.nan)
    s12, c12 = d.rsum(logretx, 12), d.rcount(retx, 12)
    ch["nissue12"] = np.where(c12 == 12, lme - d.lag(lme, 12) - s12, np.nan)
    s36, c36 = d.rsum(logret, 36), d.rcount(ret, 36)
    ch["cissue36"] = np.where(c36 == 36, lme - d.lag(lme, 36) - s36, np.nan)
    divret = ret_crsp - retx
    ch["dy12"] = np.where(d.rcount(divret, 12) >= 10, d.rsum(divret, 12), np.nan)
    ct = d.rcount(turn, 12)
    with np.errstate(invalid="ignore", divide="ignore"):
        ch["turn12"] = np.where(ct >= 9, d.rsum(turn, 12) / ct, np.nan)
    ch["maxret12"] = np.where(d.rcount(ret, 12) >= 9, d.rmax(ret, 12, 1), np.nan)
    li = d.gcumsum(logretx)            # log price index; a missing month carries the previous level
    ch["hi12"] = np.where(c12 >= 9, li - d.rmax(li, 12, 1), np.nan)

    # ---- candidates --------------------------------------------------------------------------------------
    cand = in_panel & common & (me > 0) & (prc > 0)
    out = pd.DataFrame({"PERMNO": permno[cand].astype(np.int64), "YYYYMM": yyyymm[cand].astype(np.int64),
                        "mi": mi[cand].astype(np.int64), "me": me[cand], "exch": exch[cand].astype(str),
                        "sic2": sic2[cand]})
    for c in CHARS:
        v = ch[c][cand]
        out[c] = np.where(np.isfinite(v), v, np.nan).astype("float32")
    return out, gaps


def main() -> None:
    paths.log_to("build_characteristics.txt")
    t0 = time.time()
    print(f"[input  {paths.CRSP_PANEL}]\n[input  {paths.FACTORS}]\n[output {paths.PROCESSED / OUT_NAME}]\n")
    con = duckdb.connect()
    p = con.sql(f"""
        SELECT PERMNO, YYYYMM, mi, ret, ret_crsp, retx, me, prc, turnover, common, exch, sic
        FROM read_parquet({paths.sql_path(paths.CRSP_PANEL)}) ORDER BY PERMNO, mi""").df()
    f = con.sql(f"SELECT YYYYMM, rf, mktrf, smb, hml FROM read_parquet({paths.sql_path(paths.FACTORS)})").df()
    f["mi"] = (f.YYYYMM // 100) * 12 + f.YYYYMM % 100 - 1
    print(f"panel: {len(p):,} rows, {p.PERMNO.nunique():,} PERMNOs, {p.YYYYMM.min()}-{p.YYYYMM.max()}")
    if p.duplicated(["PERMNO", "mi"]).any():
        sys.exit("ERROR: duplicate PERMNO-months in the panel; build_panel.py should have removed them.")
    need = p.mi.min(), p.mi.max()
    have = f.dropna(subset=["rf", "mktrf", "smb", "hml"]).mi
    if have.min() > need[0] or have.max() < need[1]:
        print("WARNING: the factor file does not cover every panel month; regressions there will be missing.")

    r_ind, m6 = industry_tables(p)
    permnos = p.PERMNO.unique()
    pieces, gaps = [], 0
    for k in range(0, len(permnos), CHUNK_PERMNOS):     # stock by stock in chunks: keeps memory low
        sub = p[p.PERMNO.isin(permnos[k:k + CHUNK_PERMNOS])]
        piece, gp = chunk_characteristics(sub, f, r_ind, m6)
        pieces.append(piece)
        gaps += gp
    out = pd.concat(pieces, ignore_index=True).sort_values(["mi", "PERMNO"]).reset_index(drop=True)
    print(f"gap months filled with NaN inside stock histories: {gaps:,}")
    duckdb.from_df(out).write_parquet(str(paths.PROCESSED / OUT_NAME))

    print(f"\n[wrote {OUT_NAME}: {len(out):,} candidate PERMNO-months in {time.time() - t0:.0f}s]")
    print("\nShare of candidates with a non-missing value, by decade of tau:")
    cov = out.assign(decade=(out.YYYYMM // 1000) * 10).groupby("decade")[CHARS].agg(lambda s: s.notna().mean())
    print(cov.T.round(2).to_string())
    first_full = out.dropna(subset=CHARS).YYYYMM.min()
    print(f"\nFirst month in which some stock has all 18 characteristics: {first_full}")
    print("\nMedian cross-sectional Spearman correlations with size and mom12_2 (December cross-sections):")
    sample = out[out.YYYYMM % 100 == 12]
    cors = sample.groupby("YYYYMM")[CHARS].apply(lambda x: x.rank().corr().loc[["size", "mom12_2"]])
    print(cors.groupby(level=1).median().round(2).to_string())


def d_lag_X(d: Dense, X: np.ndarray, k: int) -> np.ndarray:
    """Lag every column of X by k months (the constant column stays 1 where the lag exists)."""
    return np.column_stack([d.lag(X[:, j], k) for j in range(X.shape[1])])


if __name__ == "__main__":
    main()