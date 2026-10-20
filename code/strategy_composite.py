"""
strategy_composite.py: real-time, cost-aware composite of 18 CRSP-only characteristics (specification S-01).

Usage (from the project folder):
    python code/build_characteristics.py          # once, creates data/processed/characteristics.parquet
    python code/strategy_composite.py             # primary specification
    python code/strategy_composite.py --weighting ew          # pre-registered secondary: equal weights
    python code/strategy_composite.py --spreads path.parquet  # spread file (default: see SPREADS below)

Each month tau (end of month), using only information available at that date:
  1. Cross-section: candidates from characteristics.parquet with at least MIN_CHARS of the 18 characteristics.
     Each characteristic is ranked across all candidates and scaled to [-0.5, 0.5]; a missing value gets the
     cross-sectional median rank, 0 (a disclosed choice: the stock is treated as average on that dimension;
     the raw value is never set to zero).
  2. Weights (Fama-MacBeth, Lewellen 2015 style): each month, regress next-month excess returns on the ranks
     (stocks at or above the NYSE 20th size percentile, so microcaps do not drive the weights). The forecast at
     tau uses the average of ALL slopes whose return month is <= tau, and needs at least MIN_SLOPES of them.
  3. Portfolios: value-weighted, NYSE breakpoints of the forecast, with a buy/hold band (Novy-Marx & Velikov
     2016): a stock enters the long (short) leg in the top (bottom) ENTER share of forecasts and stays until it
     leaves the top (bottom) EXIT share. Unbanded deciles are reported as robustness.
  4. Trading costs: professor's formula TC = sum_i |traded weight_i| x SPREAD_i / 2, both legs, with traded
     weights measured against drifted weights (context file section 7b). Costs of rebalancing at the end of tau
     are charged to the return of month tau + 1. Without a spread file the script reports gross results,
     turnover and the break-even spread instead of net results.
  5. Evaluation with evaluate.py: alphas against CAPM ... q5, OLS t; headline = FF6 alpha of the NET banded
     long-short return, Jan 1974 - Dec 2025, |t| >= 2.78.

Spread file (when available): data/processed/spreads_monthly.parquet with columns PERMNO, YYYYMM, spread,
where spread is the FULL proportional bid-ask spread in decimals (0.01 = 1%) for trades at the end of YYYYMM.
A traded stock without a spread gets the median spread of its NYSE size decile that month (count reported).

Outputs (output/tables/):
  composite_returns.csv     monthly returns of every series, turnover, number of stocks
  composite_alphas.csv      alpha tables (all models) for every series and subperiod
  composite_slopes.csv      the monthly Fama-MacBeth slopes (diagnostic)
  data/processed/composite_forecasts.parquet   forecast and leg membership per PERMNO-month (for later tables)
  Log: output/logs/strategy_composite.txt

CHANGING ANY CONSTANT BELOW CREATES A NEW SPECIFICATION: log it in section 12 of context_ai_project.md first.
"""
from __future__ import annotations

import argparse
import time

import duckdb
import numpy as np
import pandas as pd

import paths
from evaluate import alpha_table, load_factors

SPEC_ID = "S-01"
CHARS = ["size", "price", "rev1", "mom12_2", "mom12_7", "ltrev36_13", "season", "ind_mom6", "ind_rel1",
         "resmom", "nissue12", "cissue36", "dy12", "beta36", "ivol36", "turn12", "maxret12", "hi12"]
MIN_CHARS = 9               # a stock needs at least 9 of the 18 characteristics
FORMATION_START = 197012    # first month whose characteristics enter a Fama-MacBeth regression
MIN_SLOPES = 36             # forecasts need at least 36 past monthly slopes
EST_SIZE_PCTL = 0.20        # estimation sample: market cap >= NYSE 20th percentile
ENTER, EXIT = 0.10, 0.20    # buy/hold band on NYSE breakpoints of the forecast
TEST_START, TEST_END = 197401, 202512   # primary test window (holding months)
SUBPERIODS = [(197401, 200012), (200101, 202512)]   # pre/post decimalization
SPREADS = paths.PROCESSED / "spreads_monthly.parquet"


def to_mi(yyyymm):
    return (np.asarray(yyyymm) // 100) * 12 + np.asarray(yyyymm) % 100 - 1


def to_yyyymm(mi):
    mi = np.asarray(mi)
    return (mi // 12) * 100 + mi % 12 + 1


def scaled_ranks(df: pd.DataFrame) -> np.ndarray:
    """Cross-sectional ranks scaled to [-0.5, 0.5] per month; missing -> 0 (the median)."""
    out = np.zeros((len(df), len(CHARS)))
    g = df.groupby("mi")
    for j, c in enumerate(CHARS):            # one column at a time keeps memory low
        r = g[c].rank(method="average").to_numpy(dtype=float, na_value=np.nan)
        n = g[c].transform("count").to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(n > 1, (r - 1) / (n - 1) - 0.5, 0.0)
        out[:, j] = np.where(np.isnan(s), 0.0, s)
    return out


def leg_return(members: np.ndarray, w_base: np.ndarray, r_next: np.ndarray):
    """Weighted next-month return of a leg; stocks with a missing return are dropped and weights renormalized."""
    w = np.where(members, w_base, 0.0)
    ok = members & ~np.isnan(r_next)
    if w.sum() <= 0 or not ok.any():
        return np.nan, w / max(w.sum(), 1e-300), int((members & np.isnan(r_next)).sum())
    w_target = w / w.sum()
    w_ok = np.where(ok, w, 0.0)
    return float((w_ok * np.nan_to_num(r_next)).sum() / w_ok.sum()), w_target, int((members & ~ok).sum())


def traded_weights(prev: pd.Series | None, prev_ret: pd.Series | None, new: pd.Series) -> pd.Series:
    """|w_new - w_drifted| per stock. prev: target weights set one month earlier; prev_ret: their realized
    returns since then (a missing return is treated as 0 for the drift only); new: today's target weights."""
    if prev is None or prev.sum() == 0:
        return new.abs()
    v = prev * (1 + prev_ret.reindex(prev.index).fillna(0.0))
    drift = v / v.sum()
    allidx = drift.index.union(new.index)
    return (new.reindex(allidx, fill_value=0.0) - drift.reindex(allidx, fill_value=0.0)).abs()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weighting", choices=["vw", "ew"], default="vw")
    ap.add_argument("--spreads", default=None, help=f"spread file (default {SPREADS.name} if it exists)")
    a = ap.parse_args()
    tag = "" if a.weighting == "vw" else "_ew"
    paths.log_to(f"strategy_composite{tag}.txt")
    t0 = time.time()
    print(f"Specification {SPEC_ID}{tag}: weighting={a.weighting}, MIN_CHARS={MIN_CHARS}, "
          f"FORMATION_START={FORMATION_START}, MIN_SLOPES={MIN_SLOPES}, EST_SIZE_PCTL={EST_SIZE_PCTL}, "
          f"band {ENTER:.0%}/{EXIT:.0%}, test {TEST_START}-{TEST_END}\n")

    # ---- data --------------------------------------------------------------------------------------------
    con = duckdb.connect()
    ch_path = paths.PROCESSED / "characteristics.parquet"
    df = con.sql(f"""
        WITH c AS (SELECT * FROM read_parquet({paths.sql_path(ch_path)})),
             nxt AS (SELECT PERMNO, mi - 1 AS mi, ret AS ret_next
                     FROM read_parquet({paths.sql_path(paths.CRSP_PANEL)}) WHERE eligible)
        SELECT c.*, nxt.ret_next FROM c LEFT JOIN nxt USING (PERMNO, mi)
        ORDER BY mi, PERMNO""").df()
    fac = load_factors()
    rf = fac["rf"]
    df["n_chars"] = df[CHARS].notna().sum(axis=1)
    df = df[df.n_chars >= MIN_CHARS].reset_index(drop=True)
    df["rf_next"] = rf.reindex(to_yyyymm(df.mi.to_numpy() + 1)).to_numpy()
    R = scaled_ranks(df)
    df = df.drop(columns=CHARS)          # raw values no longer needed (memory)
    mi_arr = df.mi.to_numpy()
    bounds = np.r_[0, np.flatnonzero(np.diff(mi_arr)) + 1, len(df)]
    months = mi_arr[bounds[:-1]]
    print(f"cross-sections: {len(df):,} PERMNO-months with >= {MIN_CHARS} characteristics, "
          f"{to_yyyymm(months[0])}-{to_yyyymm(months[-1])}")

    spreads = None
    sp_path = a.spreads or (SPREADS if SPREADS.exists() else None)
    if sp_path:
        spreads = con.sql(f"SELECT PERMNO, YYYYMM, spread FROM read_parquet({paths.sql_path(sp_path)})").df()
        spreads = spreads.set_index(["PERMNO", "YYYYMM"]).spread
        print(f"spreads: {sp_path} ({len(spreads):,} PERMNO-months)")
    else:
        print("spreads: NONE -> gross results, turnover and break-even spread only (no net returns)")

    me_all = df.me.to_numpy()
    nyse_all = (df.exch.to_numpy() == "N")
    rn_all = df.ret_next.to_numpy()
    y_all = rn_all - df.rf_next.to_numpy()
    perm_all = df.PERMNO.to_numpy()

    # ---- 1. monthly Fama-MacBeth slopes ------------------------------------------------------------------
    slopes, n_est = {}, {}
    form_start_mi = to_mi(FORMATION_START)
    for j, m in enumerate(months):
        if m < form_start_mi:
            continue
        sl = slice(bounds[j], bounds[j + 1])
        me, ny, y = me_all[sl], nyse_all[sl], y_all[sl]
        if ny.sum() < 20:
            continue
        p20 = np.quantile(me[ny], EST_SIZE_PCTL)
        sel = (me >= p20) & ~np.isnan(y)
        if sel.sum() < 50:
            continue
        X = np.column_stack([np.ones(sel.sum()), R[sl][sel]])
        b, *_ = np.linalg.lstsq(X, y[sel], rcond=None)
        slopes[m + 1] = b                     # known at the end of month m + 1 (its return month)
        n_est[m + 1] = int(sel.sum())
    S = pd.DataFrame.from_dict(slopes, orient="index", columns=["const"] + CHARS).sort_index()
    avg = S.expanding().mean()
    cnt = S["const"].expanding().count()
    print(f"Fama-MacBeth: {len(S)} monthly regressions, return months {to_yyyymm(S.index[0])}-"
          f"{to_yyyymm(S.index[-1])}, median {int(np.median(list(n_est.values())))} stocks each")

    # ---- 2. forecasts, portfolios, turnover, costs -------------------------------------------------------
    out, fc_rows = [], []
    state = {k: {"w": None, "ret": None} for k in ("L", "S", "L0", "S0")}
    prev_long, prev_short = pd.Index([]), pd.Index([])
    cost_imputed, cost_total = 0, 0
    for j, m in enumerate(months):
        if m not in avg.index or cnt.loc[m] < MIN_SLOPES:     # slopes with return month <= m
            continue
        sl = slice(bounds[j], bounds[j + 1])
        f = R[sl] @ avg.loc[m, CHARS].to_numpy()
        perm, me, ny, rn = perm_all[sl], me_all[sl], nyse_all[sl], rn_all[sl]
        q10, q20, q80, q90 = np.quantile(f[ny], [ENTER, EXIT, 1 - EXIT, 1 - ENTER])
        was_long, was_short = np.isin(perm, prev_long), np.isin(perm, prev_short)
        long_b = (f >= q90) | (was_long & (f >= q80))
        short_b = (f <= q10) | (was_short & (f <= q20))
        long_0, short_0 = f >= q90, f <= q10
        prev_long, prev_short = pd.Index(perm[long_b]), pd.Index(perm[short_b])
        w_base = me if a.weighting == "vw" else np.ones(len(me))
        rec = {"YYYYMM": int(to_yyyymm(m + 1))}
        # NYSE size deciles (for spread imputation)
        size_dec = np.searchsorted(np.quantile(me[ny], np.linspace(0.1, 0.9, 9)), me, side="right")
        for leg, members in (("L", long_b), ("S", short_b), ("L0", long_0), ("S0", short_0)):
            r, w_t, n_miss = leg_return(members, w_base, rn)
            w_new = pd.Series(w_t[members], index=perm[members])
            st = state[leg]
            tw = traded_weights(st["w"], st["ret"], w_new)
            rec[f"r_{leg}"], rec[f"to_{leg}"], rec[f"n_{leg}"] = r, tw.sum(), int(members.sum())
            rec[f"miss_{leg}"] = n_miss
            if spreads is not None:
                key = pd.MultiIndex.from_arrays([tw.index.to_numpy(), np.full(len(tw), int(to_yyyymm(m)))])
                s = pd.Series(spreads.reindex(key).to_numpy(), index=tw.index)
                if s.isna().any():   # impute by NYSE size decile this month
                    month_sp = pd.Series(spreads.reindex(pd.MultiIndex.from_arrays(
                        [perm, np.full(len(perm), int(to_yyyymm(m)))])).to_numpy(), index=perm)
                    dec = pd.Series(size_dec, index=perm)
                    med = month_sp.groupby(dec).median()
                    miss = s.index[s.isna()]
                    fill = dec.reindex(miss).map(med)
                    # stocks no longer in the cross-section (e.g. delisted) have no decile: month median
                    fill = fill.fillna(month_sp.median())
                    cost_imputed += int(fill.notna().sum())
                    s.loc[miss] = fill.to_numpy()
                cost_total += len(tw)
                rec[f"tc_{leg}"] = float((tw * s / 2).sum()) if s.notna().all() else np.nan
            st["w"], st["ret"] = w_new, pd.Series(rn[members], index=perm[members])
        out.append(rec)
        fc_rows.append(pd.DataFrame({"PERMNO": perm.astype(np.int32), "YYYYMM": np.int32(to_yyyymm(m)),
                                     "forecast": f.astype(np.float32), "long": long_b, "short": short_b,
                                     "ret_next": rn.astype(np.float32)}))
    res = pd.DataFrame(out).set_index("YYYYMM").sort_index()
    res["ls_gross"] = res.r_L - res.r_S
    res["ls_gross_noband"] = res.r_L0 - res.r_S0
    res["long_gross"] = res.r_L
    if spreads is not None:
        res["ls_net"] = res.ls_gross - res.tc_L - res.tc_S
        res["ls_net_noband"] = res.ls_gross_noband - res.tc_L0 - res.tc_S0
        res["long_net"] = res.r_L - res.tc_L
    fcs = pd.concat(fc_rows, ignore_index=True)
    duckdb.from_df(fcs).write_parquet(str(paths.PROCESSED / f"composite_forecasts{tag}.parquet"))

    # ---- 3. evaluation -----------------------------------------------------------------------------------
    paths.ensure_dirs()
    win = res.loc[TEST_START:TEST_END]
    print(f"\nPortfolios: holding months {res.index[0]}-{res.index[-1]}; primary window {TEST_START}-"
          f"{TEST_END} = {len(win)} months")
    print(f"stocks per leg (median): long {win.n_L.median():.0f}, short {win.n_S.median():.0f}; "
          f"min {win.n_L.min():.0f} / {win.n_S.min():.0f}")
    print(f"members dropped for a missing next-month return (share of leg-months): "
          f"{(win.miss_L.sum() + win.miss_S.sum()) / (win.n_L.sum() + win.n_S.sum()):.4f}")
    to = win[["to_L", "to_S", "to_L0", "to_S0"]].mean()
    print("\nTraded weight per month, sum |w_new - w_drifted| (one-way turnover = half of it):")
    print(f"  banded:   long {to.to_L:.3f}, short {to.to_S:.3f}")
    print(f"  unbanded: long {to.to_L0:.3f}, short {to.to_S0:.3f}")
    if spreads is not None:
        print(f"  traded stock-months with an imputed spread: {cost_imputed:,} of {cost_total:,}")

    series = ["ls_net", "ls_gross", "ls_net_noband", "ls_gross_noband", "long_net", "long_gross"]
    series = [s for s in series if s in res]
    pd.set_option("display.width", 220)
    tables = []
    show = ["alpha_pct", "t_ols", "r2", "n", "first", "last", "ann_appraisal", "passes_2.78"]
    for s in series:
        y = win[s].dropna()
        tab = alpha_table(y, fac, long_only=s.startswith("long"))
        print(f"\n=== {s}: mean {100 * y.mean():.3f}% per month, t(mean) "
              f"{y.mean() / y.std() * np.sqrt(len(y)):.2f}, {len(y)} months ===")
        print(tab[show].round(3).to_string())
        tables.append(tab.assign(series=s, window=f"{TEST_START}-{TEST_END}"))
        if s in ("ls_net", "ls_gross"):
            for lo, hi in SUBPERIODS:
                ys = res.loc[lo:hi, s].dropna()
                ts = alpha_table(ys, fac)
                print(f"   {s} {lo}-{hi}: FF6 alpha {ts.loc['FF6', 'alpha_pct']:.3f}% "
                      f"(t {ts.loc['FF6', 't_ols']:.2f}, {len(ys)} months)")
                tables.append(ts.assign(series=s, window=f"{lo}-{hi}"))
    headline = "ls_net" if "ls_net" in res else "ls_gross"
    hl = [t for t in tables if t.series.iloc[0] == headline][0].loc["FF6"]
    # The sign is pre-registered (long high forecasts), so the test is one-sided: t must be >= +2.78.
    verdict = ("PASSES t >= 2.78" if hl.t_ols >= 2.78 else
               "significantly NEGATIVE (t <= -2.78): fails" if hl.t_ols <= -2.78 else "does not pass t >= 2.78")
    print(f"\nHEADLINE ({headline}, FF6, {TEST_START}-{TEST_END}): alpha {hl.alpha_pct:.3f}% per month, "
          f"OLS t = {hl.t_ols:.2f} -> {verdict}")
    if headline == "ls_gross":
        print("  (GROSS only: the hurdle applies net of costs; add a spread file to get the real headline)")
        tw = (to.to_L + to.to_S)
        g = [t for t in tables if t.series.iloc[0] == "ls_gross"][0].loc["FF6", "alpha_pct"] / 100
        if g > 0:
            print(f"  Break-even uniform spread (professor's formula, both legs): 2 x gross FF6 alpha / traded "
                  f"weight = {200 * g / tw:.2f}% full spread (traded weight {tw:.3f} per month)")
        else:
            print("  No positive gross FF6 alpha, so there is no break-even spread.")

    # ---- 4. diagnostics ----------------------------------------------------------------------------------
    Sw = S.loc[to_mi(TEST_START):]                     # return months of the test window
    diag = pd.DataFrame({"mean_slope_pct": 100 * Sw[CHARS].mean(),
                         "fm_t": Sw[CHARS].mean() / Sw[CHARS].std() * np.sqrt(Sw[CHARS].count()),
                         "share_positive": (Sw[CHARS] > 0).mean(),
                         "weight_at_end_pct": 100 * avg[CHARS].iloc[-1]})
    print("\nFama-MacBeth slopes on scaled ranks over the test window (DESCRIPTIVE ONLY; the strategy uses "
          "real-time averages):")
    print(diag.round(3).to_string())
    ic = fcs.dropna(subset=["ret_next"]).groupby("YYYYMM")[["forecast", "ret_next"]].apply(
        lambda x: x.forecast.rank().corr(x.ret_next.rank()))
    ic = ic.loc[ic.index >= to_yyyymm(to_mi(TEST_START) - 1)]   # formation months of the test window
    print(f"\nRank correlation of forecast with next-month return: mean {ic.mean():.4f}, "
          f"t {ic.mean() / ic.std() * np.sqrt(len(ic)):.2f}, {len(ic)} months")

    res.to_csv(paths.TABLES / f"composite_returns{tag}.csv")
    pd.concat(tables).to_csv(paths.TABLES / f"composite_alphas{tag}.csv")
    S.set_index(pd.Index(to_yyyymm(S.index), name="return_month")).to_csv(paths.TABLES / f"composite_slopes{tag}.csv")
    print(f"\n[wrote composite_returns{tag}.csv, composite_alphas{tag}.csv, composite_slopes{tag}.csv to "
          f"{paths.TABLES}; composite_forecasts{tag}.parquet to {paths.PROCESSED}; {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()