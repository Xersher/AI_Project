"""
plot_alphas.py: how the strategy's alpha evolves over time, for the composite and for each of its
18 characteristics on their own.

Usage (from the project folder), after build_characteristics.py and strategy_composite.py:
    python code/plot_alphas.py                     # FF6, 60-month rolling windows
    python code/plot_alphas.py --model q5          # any model of evaluate.py: CAPM FF3 FF3+Mom FF5 FF6 q q5
    python code/plot_alphas.py --window 120        # 10-year rolling windows
    python code/plot_alphas.py --sign realtime     # sign each characteristic the way the composite traded it
    python code/plot_alphas.py --tag ew            # the equal-weighted run (composite_returns_ew.csv)
    python code/plot_alphas.py --note "Draft"      # extra line under every title (e.g. "draft, not final")

Reads
  output/tables/composite_returns{tag}.csv       strategy_composite.py
  output/tables/composite_slopes{tag}.csv        strategy_composite.py (only for --sign realtime)
  data/processed/characteristics.parquet, crsp_panel.parquet, factors_monthly.parquet
Writes (figures as PNG and PDF; every plotted number is also saved as CSV, the figures' table view;
        with --sign realtime the individual outputs get the suffix _realtime)
  output/figures/alphas/composite_<series>_<model>.*         the total alpha: three panels (a)-(c) below
  output/figures/alphas/individual/<characteristic>_<model>.* the same three panels for each characteristic
  output/figures/alphas/individual_grid_rolling_<model>.*     18 small multiples of panel (b), one shared y-axis
  output/figures/alphas/individual_grid_cumulative_<model>.*  18 small multiples of panel (a), one shared y-axis
  output/tables/alpha_paths_<model>{tag}.csv                  every plotted point, all series
  output/tables/individual_returns{tag}.csv                   monthly returns of the 18 single-characteristic portfolios
  output/tables/individual_alphas_<model>{tag}.csv            their full-sample alpha tables (all models)

The three panels (window = TEST_START..TEST_END of strategy_composite.py, Jan 1974 - Dec 2025):
  (a) Cumulative abnormal return: the monthly return minus its factor exposure, b'f_t, with b estimated once
      on the whole window, summed in %. Its slope is the alpha; flat stretches are periods without alpha.
      Because OLS residuals sum to zero, the line ends exactly at alpha x N. The betas use the whole window,
      so this is a description of the sample, not a tradable series.
  (b) Rolling alpha: OLS alpha over the WINDOW months ending in the month shown, with a pointwise 95% band
      (+-1.96 OLS standard errors). Windows overlap, so neighbouring points are not independent.
  (c) Expanding t-statistic: OLS t of the alpha using every month from Jan 1974 to the month shown (shown from
      the WINDOW-th month on), with the +-2.78 hurdle. The last point is the full-sample t.
A vertical hairline marks Jan 2001, the pre-registered subperiod split.

Single-characteristic portfolios ("individual alphas"): each month tau, the stocks in the composite's universe
(at least MIN_CHARS of the 18 characteristics) that have a value for the characteristic are sorted into deciles
on NYSE breakpoints; the portfolio earns the month tau+1 return of decile 10 minus decile 1, value-weighted by
market cap at tau (equal-weighted with --tag ew). Breakpoints never use next month's return. Returns are GROSS
(no costs, no band): they show where the composite's raw predictability comes from, not tradable results.
  --sign raw (default): always high minus low, the convention of the literature; a negative alpha means
                        low values of the characteristic earn more (e.g. short-term reversal).
  --sign realtime:      each month the spread is multiplied by the sign of the composite's real-time weight on
                        that characteristic (average of past Fama-MacBeth slopes), so a positive alpha means the
                        composite's choice of direction paid off. Starts when the composite starts trading.

Needs matplotlib (pip install matplotlib). Changing --window or --model does not change the strategy, but report
the choice with every figure.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402
from evaluate import MODELS, alpha_table, load_factors, ols  # noqa: E402
from strategy_composite import (CHARS, MIN_CHARS, MIN_SLOPES, SUBPERIODS, TEST_END, TEST_START,  # noqa: E402
                                to_mi, to_yyyymm)

HURDLE = 2.78
MIN_LEG = 5                 # a decile needs at least 5 stocks with a return, otherwise the month is missing
SPLIT = SUBPERIODS[1][0]    # 200101

NAMES = {"size": "Size (log market cap)", "price": "Price (log)", "rev1": "Short-term reversal (t-1)",
         "mom12_2": "Momentum 12-2", "mom12_7": "Momentum 12-7", "ltrev36_13": "Long-term reversal 36-13",
         "season": "Seasonality", "ind_mom6": "Industry momentum 6m", "ind_rel1": "Industry-relative return t-1",
         "resmom": "Residual momentum", "nissue12": "Net issuance 12m", "cissue36": "Composite issuance 36m",
         "dy12": "Dividend yield 12m", "beta36": "Market beta 36m", "ivol36": "Idiosyncratic volatility 36m",
         "turn12": "Share turnover 12m", "maxret12": "Max monthly return 12m", "hi12": "Distance to 12m high"}
SERIES_NAMES = {"ls_net": "Composite, net of costs (banded long-short)",
                "ls_gross": "Composite, gross (banded long-short)",
                "ls_net_noband": "Composite, net of costs (unbanded long-short)",
                "ls_gross_noband": "Composite, gross (unbanded long-short)",
                "long_net": "Composite long leg, net of costs (vs. T-bills)",
                "long_gross": "Composite long leg, gross (vs. T-bills)"}

# Colours: one series per panel, so one hue (validated: passes all checks on a white surface).
BLUE = "#2a78d6"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 9, "axes.titlelocation": "left", "axes.titlepad": 6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "xtick.color": MUTED,
        "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.size": 3,
        "ytick.major.size": 0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "savefig.dpi": 200, "pdf.fonttype": 42})


def dates(yyyymm) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(yyyymm).astype(int).astype(str), format="%Y%m")


# ---------------------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------------------
def alpha_paths(y: pd.Series, fac: pd.DataFrame, model: str, window: int, long_only: bool) -> tuple:
    """Cumulative abnormal return, rolling alpha and SE, expanding t, plus the full-sample summary."""
    cols = MODELS[model]
    yy = y.astype(float) - (fac["rf"].reindex(y.index) if long_only else 0.0)
    d = pd.concat([yy.rename("y"), fac[cols]], axis=1, join="inner").dropna()
    d = d.loc[TEST_START:TEST_END]
    out = pd.DataFrame(index=d.index)
    if len(d) < max(window, 24):
        return None, None
    Y, X = d["y"].to_numpy(), d[cols].to_numpy()
    full = ols(Y, X)
    out["cum_abnormal_pct"] = 100 * np.cumsum(Y - X @ full["b"][1:])
    roll_a = np.full(len(d), np.nan)
    roll_se = np.full(len(d), np.nan)
    exp_t = np.full(len(d), np.nan)
    contiguous = np.diff(to_mi(d.index.to_numpy()), prepend=to_mi(d.index[0]) - 1) == 1
    for i in range(window - 1, len(d)):
        if contiguous[i - window + 1:i + 1].all():           # only windows without missing months
            r = ols(Y[i - window + 1:i + 1], X[i - window + 1:i + 1])
            roll_a[i], roll_se[i] = r["b"][0], r["b"][0] / r["t"][0]
        exp_t[i] = ols(Y[:i + 1], X[:i + 1])["t"][0]
    out["rolling_alpha_pct"] = 100 * roll_a
    out["rolling_se_pct"] = 100 * roll_se
    out["expanding_t"] = exp_t
    summary = {"alpha_pct": 100 * full["b"][0], "t": full["t"][0], "n": full["n"],
               "first": int(d.index[0]), "last": int(d.index[-1])}
    return out, summary


def single_characteristic_returns(weighting: str, sign: str, tag: str) -> pd.DataFrame:
    """Monthly decile-10-minus-decile-1 returns for each characteristic (holding month index)."""
    con = duckdb.connect()
    df = con.sql(f"""
        WITH c AS (SELECT * FROM read_parquet({paths.sql_path(paths.PROCESSED / "characteristics.parquet")})),
             nxt AS (SELECT PERMNO, mi - 1 AS mi, ret AS ret_next
                     FROM read_parquet({paths.sql_path(paths.CRSP_PANEL)}) WHERE eligible)
        SELECT c.mi, c.me, c.exch, {", ".join("c." + ch for ch in CHARS)}, nxt.ret_next
        FROM c LEFT JOIN nxt USING (PERMNO, mi) ORDER BY c.mi""").df()
    df = df[df[CHARS].notna().sum(axis=1) >= MIN_CHARS]          # the composite's universe
    mi_arr = df.mi.to_numpy()
    V = df[CHARS].to_numpy(dtype=float)
    me, ny, rn = df.me.to_numpy(float), df.exch.to_numpy() == "N", df.ret_next.to_numpy(float)
    bounds = np.r_[0, np.flatnonzero(np.diff(mi_arr)) + 1, len(df)]
    rows = {}
    for j in range(len(bounds) - 1):
        sl = slice(bounds[j], bounds[j + 1])
        w = me[sl] if weighting == "vw" else np.ones(bounds[j + 1] - bounds[j])
        r, nys = rn[sl], ny[sl]
        rec = {}
        for k, c in enumerate(CHARS):
            x = V[sl, k]
            has = ~np.isnan(x)
            if (has & nys).sum() < 20:
                rec[c] = np.nan
                continue
            q10, q90 = np.quantile(x[has & nys], [0.1, 0.9])
            legs = []
            for members in (has & (x >= q90), has & (x <= q10)):
                ok = members & ~np.isnan(r)
                legs.append((w[ok] * r[ok]).sum() / w[ok].sum() if ok.sum() >= MIN_LEG else np.nan)
            rec[c] = legs[0] - legs[1]
        rows[int(to_yyyymm(mi_arr[bounds[j]] + 1))] = rec            # holding month
    out = pd.DataFrame.from_dict(rows, orient="index")[CHARS].sort_index()
    out.index.name = "YYYYMM"
    if sign == "realtime":
        f = paths.TABLES / f"composite_slopes{tag}.csv"
        if not f.exists():
            raise SystemExit(f"--sign realtime needs {f}; run strategy_composite.py first.")
        S = pd.read_csv(f, index_col=0)
        S.index = to_mi(S.index.to_numpy())
        full = range(int(S.index.min()), int(S.index.max()) + 1)
        avg = S[CHARS].expanding().mean().reindex(full).ffill()
        cnt = S["const"].expanding().count().reindex(full).ffill()
        form = to_mi(out.index.to_numpy()) - 1                           # formation month of each return
        sg = np.sign(avg.reindex(form).to_numpy())
        sg[(cnt.reindex(form).to_numpy() < MIN_SLOPES) | np.isnan(cnt.reindex(form).to_numpy())] = np.nan
        out = out * sg
    return out


# ---------------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------------
def decorate_time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(10, month=1, day=1))
    ax.xaxis.set_minor_locator(mdates.YearLocator(5, month=1, day=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", which="minor", length=2, color=AXIS)
    ax.axvline(dates([SPLIT])[0], color=AXIS, lw=0.8, zorder=1)


def baseline(ax) -> None:
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)


def right_label(ax, y, text, va="center") -> None:
    ax.annotate(text, xy=(1, y), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                color=MUTED, fontsize=7.5, va=va, ha="left", annotation_clip=False)


def three_panel(p: pd.DataFrame, s: dict, title: str, model: str, window: int, note: str, out: Path,
                formats: list[str]) -> None:
    x = dates(p.index)
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 7.6), sharex=True, gridspec_kw={"hspace": 0.42})
    sub = (f"{model} alpha {s['alpha_pct']:.3f}% per month, OLS t = {s['t']:.2f}, N = {s['n']} months "
           f"({s['first']}-{s['last']})")
    fig.text(0.075, 0.975, title, color=INK, fontsize=10.5, fontweight="bold", va="top")
    fig.text(0.075, 0.948, sub + (f"\n{note}" if note else ""), color=INK2, fontsize=8.5, va="top")

    ax = axs[0]
    ax.plot(x, p.cum_abnormal_pct, color=BLUE, lw=1.5, zorder=3)
    ax.plot(x[-1:], p.cum_abnormal_pct.iloc[-1:], "o", ms=5, color=BLUE, mec=SURFACE, mew=1.5, zorder=4)
    right_label(ax, p.cum_abnormal_pct.iloc[-1], f"{p.cum_abnormal_pct.iloc[-1]:.0f}%")
    baseline(ax)
    ax.set_title(f"(a) Cumulative {model} abnormal return, % (slope = alpha; full-window betas)")
    ax.annotate(str(SPLIT // 100), xy=(dates([SPLIT])[0], 1), xycoords=("data", "axes fraction"),
                xytext=(3, -2), textcoords="offset points", color=MUTED, fontsize=7.5, va="top")

    ax = axs[1]
    a, se = p.rolling_alpha_pct, p.rolling_se_pct
    ax.fill_between(x, a - 1.96 * se, a + 1.96 * se, color=BLUE, alpha=0.12, lw=0, zorder=2)
    ax.plot(x, a, color=BLUE, lw=1.5, zorder=3)
    baseline(ax)
    ax.axhline(s["alpha_pct"], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    right_label(ax, s["alpha_pct"], f"full sample\n{s['alpha_pct']:.2f}")
    ax.set_title(f"(b) Rolling {window}-month {model} alpha, % per month (window ending in month shown), 95% band")

    ax = axs[2]
    ax.plot(x, p.expanding_t, color=BLUE, lw=1.5, zorder=3)
    baseline(ax)
    for h in (HURDLE, -HURDLE):
        ax.axhline(h, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
        right_label(ax, h, f"{h:+.2f}")
    lo, hi = np.nanmin(p.expanding_t), np.nanmax(p.expanding_t)
    ax.set_ylim(min(lo, -HURDLE) - 0.6, max(hi, HURDLE) + 0.6)
    ax.set_title(f"(c) t-statistic of the {model} alpha, all months from {TEST_START // 100} to the month shown")
    for ax in axs:
        decorate_time_axis(ax)
    fig.subplots_adjust(left=0.075, right=0.9, top=0.87 - (0.02 if note else 0), bottom=0.05)
    save(fig, out, formats)


def grid(paths_by_char: dict, summaries: dict, column: str, model: str, window: int, kind_title: str,
         note: str, out: Path, formats: list[str]) -> None:
    fig, axs = plt.subplots(6, 3, figsize=(7.5, 10.2), sharex=True, sharey=True,
                            gridspec_kw={"hspace": 0.62, "wspace": 0.12})
    for ax, c in zip(axs.ravel(), CHARS):
        p, s = paths_by_char.get(c), summaries.get(c)
        ax.set_title(NAMES[c], fontsize=8, pad=13)
        if p is None:
            ax.text(0.5, 0.5, "not enough data", transform=ax.transAxes, ha="center", color=MUTED, fontsize=7.5)
            continue
        x = dates(p.index)
        if column == "rolling_alpha_pct":
            a, se = p.rolling_alpha_pct, p.rolling_se_pct
            ax.fill_between(x, a - 1.96 * se, a + 1.96 * se, color=BLUE, alpha=0.12, lw=0, zorder=2)
        ax.plot(x, p[column], color=BLUE, lw=1.2, zorder=3)
        baseline(ax)
        ax.text(0, 1.02, f"alpha {s['alpha_pct']:.2f}%, t {s['t']:.2f}", transform=ax.transAxes,
                color=INK2, fontsize=7, va="bottom")
        decorate_time_axis(ax)
        ax.tick_params(labelsize=7)
    fig.text(0.07, 0.985, f"Single-characteristic portfolios: {kind_title}", color=INK, fontsize=10.5,
             fontweight="bold", va="top")
    unit = "% per month, 95% band" if column == "rolling_alpha_pct" else "%, slope = alpha"
    fig.text(0.07, 0.963, f"y-axis: {unit}. Decile 10 minus decile 1, NYSE breakpoints, gross of costs; "
             "one shared y-axis.\n"
             f"Full-sample {model} alpha (% per month) and OLS t above each panel." + (f"\n{note}" if note else ""),
             color=INK2, fontsize=8, va="top")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.895 - (0.012 if note else 0), bottom=0.035)
    save(fig, out, formats)


def save(fig, out: Path, formats: list[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out.with_suffix("." + fmt))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="FF6", choices=list(MODELS))
    ap.add_argument("--window", type=int, default=60, help="rolling window in months (default 60)")
    ap.add_argument("--sign", default="raw", choices=["raw", "realtime"])
    ap.add_argument("--tag", default="", help="'ew' for the equal-weighted run of strategy_composite.py")
    ap.add_argument("--series", default=None,
                    help="comma-separated composite series (default: ls_net and ls_gross, whichever exist)")
    ap.add_argument("--note", default="", help="extra line printed under every figure title")
    ap.add_argument("--formats", default="png,pdf")
    a = ap.parse_args()
    tag = f"_{a.tag}" if a.tag and not a.tag.startswith("_") else a.tag
    weighting = "ew" if tag == "_ew" else "vw"
    formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    rt = "_realtime" if a.sign == "realtime" else ""   # keeps realtime-signed outputs apart from raw ones
    paths.log_to(f"plot_alphas_{a.model}{tag}{rt}.txt")
    t0 = time.time()
    style()
    fac = load_factors()
    figdir = paths.FIGURES / "alphas"
    print(f"model {a.model}, rolling window {a.window} months, individual sign '{a.sign}', weighting {weighting}, "
          f"window {TEST_START}-{TEST_END}\n")

    # ---- total alpha: the composite ----------------------------------------------------------------------
    res = pd.read_csv(paths.TABLES / f"composite_returns{tag}.csv", index_col=0)
    wanted = a.series.split(",") if a.series else [s for s in ("ls_net", "ls_gross") if s in res]
    all_paths = []
    for s in wanted:
        if s not in res:
            raise SystemExit(f"series '{s}' is not in composite_returns{tag}.csv; columns: {list(res.columns)}")
        p, summ = alpha_paths(res[s].dropna(), fac, a.model, a.window, long_only=s.startswith("long"))
        if p is None:
            print(f"{s}: not enough months, skipped")
            continue
        three_panel(p, summ, SERIES_NAMES.get(s, s) + (" (equal-weighted)" if weighting == "ew" else ""), a.model, a.window, a.note,
                    figdir / f"composite_{s}_{a.model}{tag}", formats)
        all_paths.append(p.assign(series=s))
        print(f"{s:<16} alpha {summ['alpha_pct']:7.3f}%  t {summ['t']:6.2f}  N {summ['n']}")

    # ---- individual alphas: one portfolio per characteristic ---------------------------------------------
    ind = single_characteristic_returns(weighting, a.sign, tag)
    ind.to_csv(paths.TABLES / f"individual_returns{tag}{rt}.csv")
    by_char, summaries, tables = {}, {}, []
    print(f"\nSingle-characteristic portfolios (decile 10 minus decile 1, {weighting.upper()}, gross, sign={a.sign}):")
    for c in CHARS:
        y = ind[c].dropna()
        p, summ = alpha_paths(y, fac, a.model, a.window, long_only=False)
        if p is None:
            print(f"  {c:<11} not enough months")
            continue
        by_char[c], summaries[c] = p, summ
        three_panel(p, summ, f"{NAMES[c]}: decile 10 minus decile 1 (gross, sign = {a.sign})", a.model,
                    a.window, a.note, figdir / "individual" / f"{c}_{a.model}{tag}{rt}", formats)
        all_paths.append(p.assign(series=c))
        tables.append(alpha_table(y.loc[TEST_START:TEST_END], fac).assign(characteristic=c))
        print(f"  {c:<11} alpha {summ['alpha_pct']:7.3f}%  t {summ['t']:6.2f}  N {summ['n']}")
    grid(by_char, summaries, "rolling_alpha_pct", a.model, a.window,
         f"rolling {a.window}-month {a.model} alpha", a.note,
         figdir / f"individual_grid_rolling_{a.model}{tag}{rt}", formats)
    grid(by_char, summaries, "cum_abnormal_pct", a.model, a.window,
         f"cumulative {a.model} abnormal return", a.note,
         figdir / f"individual_grid_cumulative_{a.model}{tag}{rt}", formats)

    pd.concat(all_paths).to_csv(paths.TABLES / f"alpha_paths_{a.model}{tag}{rt}.csv")
    if tables:
        pd.concat(tables).to_csv(paths.TABLES / f"individual_alphas_{a.model}{tag}{rt}.csv")
    print(f"\n[wrote {len(wanted) + len(by_char) + 2} figures ({', '.join(formats)}) to {figdir}; "
          f"alpha_paths_{a.model}{tag}{rt}.csv, individual_returns{tag}{rt}.csv, "
          f"individual_alphas_{a.model}{tag}{rt}.csv "
          f"to {paths.TABLES}; {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()