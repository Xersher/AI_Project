"""
build_spreads.py: daily CRSP closing bid/ask quotes (large CSV) -> monthly quoted spreads per PERMNO-month.

Usage (from the project folder):
    python code/build_spreads.py --inspect     # 1 second: header, first rows, detected columns. Run this first.
    python code/build_spreads.py               # full run
Options:
    a CSV path as first argument      instead of the single .csv / .csv.gz in data/raw/spreads/
    --max-rel 0.5                     discard daily quotes whose relative spread exceeds this (default 0.5 = 50%)
    --rebuild                         re-read the CSV even if the daily Parquet file is up to date
    --memory-limit 6GB --threads 8    DuckDB resources (default: DuckDB's own, i.e. ~80% of RAM, all cores)
    --permno-col / --date-col / --bid-col / --ask-col / --date-format   override the automatic detection

What it does
  1. CSV -> data/processed/spreads_daily.parquet   (one pass over the CSV; the only slow step)
       Keeps PERMNO, date, bid, ask (the ticker is dropped: PERMNO is the key, tickers get reused).
       Duplicate PERMNO-days are merged into one row (the number merged and any disagreement are recorded).
       Re-runs reuse this file, so changing --max-rel only redoes step 2 (fast).
  2. Daily -> data/processed/spreads_monthly.parquet, one row per PERMNO-month (YYYYMM, mi as in crsp_panel):
       spread_mean    mean of the valid daily relative spreads in the month          <- recommended
       spread_last    the last valid quote of the month (never a date from the next month)
       last_date      date of that quote
       n_valid        number of valid daily quotes; n_obs = number of days in the file
     Relative spread = (ask - bid) / midpoint: the FULL quoted spread, as a decimal (0.01 = 1%).
     In the professor's formula TC = sum_i |dw_i| x SPREAD_i / 2, use it as SPREAD (do not halve it here).
     A daily quote is valid if bid > 0, ask > bid (crossed and locked quotes dropped) and spread <= --max-rel.
     Months without any valid quote get no value: missing spreads stay missing, never zero.
  3. Report (also saved to output/logs/build_spreads.txt): parsing, duplicates, quote validity by decade,
     spread levels by year, and, if crsp_panel.parquet exists, coverage of the stock universe by period and
     exchange in number of stocks and in market cap (table also saved to output/tables/).

Timing: month-t values only use quotes dated in month t, so they are known at the end of month t,
when the rebalancing trades happen.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb

import paths

RAW_SPREADS = getattr(paths, "RAW_SPREADS", paths.RAW / "spreads")
SPREADS_DAILY = getattr(paths, "SPREADS_DAILY", paths.PROCESSED / "spreads_daily.parquet")
SPREADS_MONTHLY = getattr(paths, "SPREADS_MONTHLY", paths.PROCESSED / "spreads_monthly.parquet")
META = SPREADS_DAILY.with_suffix(".meta.json")
VERSION = 1  # bump when step 1 changes, so old daily files are rebuilt

CANDIDATES = {
    "permno": ["permno"],
    "date": ["date", "dlycaldt", "caldt", "dlydt", "datadate", "yyyymmdd"],
    "bid": ["bid", "dlybid", "closebid", "closing_bid", "bid_close"],
    "ask": ["ask", "dlyask", "closeask", "closing_ask", "ask_close", "ofr", "offer"],
}
# Columns that are NOT closing quotes. Legacy CRSP BIDLO/ASKHI hold the day's LOW/HIGH trade price
# whenever the stock traded; using them as bid/ask would turn the daily range into a "spread".
NOT_QUOTES = {"bidlo", "askhi", "dlylow", "dlyhigh", "low", "high", "prclo", "prchi"}
DATE_FORMATS = ["%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%d/%m/%Y", "%d%b%Y", "%Y/%m/%d", "%d.%m.%Y"]
NULLS = ["N/A", "NA", "", ".", "NaN", "nan"]
PERIODS = [(1900, 1982), (1983, 1992), (1993, 2000), (2001, 2010), (2011, 2100)]


# ----------------------------------------------------------------------------------------------- helpers
def q(name: str) -> str:
    """Quote a column name for SQL."""
    return '"' + name.replace('"', '""') + '"'


def find_csv(arg: str | None) -> Path:
    if arg:
        p = Path(arg).expanduser().resolve()
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")
        return p
    files = sorted(RAW_SPREADS.glob("*.csv")) + sorted(RAW_SPREADS.glob("*.csv.gz")) if RAW_SPREADS.exists() else []
    if len(files) != 1:
        sys.exit(f"ERROR: expected exactly one .csv (or .csv.gz) in {RAW_SPREADS}, found {len(files)}. "
                 "Put the daily bid/ask file there, or pass its path as argument.")
    print(f"found {files[0]} in {RAW_SPREADS}")
    return files[0]


def read_head(path: Path, n: int = 3000) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    lines = []
    with opener(path, "rt", encoding="utf-8-sig", newline="") as fh:
        for line in fh:
            lines.append(line)
            if len(lines) > n:
                break
    return lines


def sniff(lines: list[str]) -> tuple[str, list[str], list[list[str]]]:
    delim = max([",", ";", "\t", "|"], key=lambda d: lines[0].count(d))
    rows = list(csv.reader(lines, delimiter=delim))
    header = [h.strip() for h in rows[0]]
    return delim, header, [r for r in rows[1:] if r]


def pick_columns(header: list[str], overrides: dict[str, str | None]) -> dict[str, str]:
    low = {h.lower().strip(): h for h in header}
    out = {}
    for role, cands in CANDIDATES.items():
        if overrides.get(role):
            if overrides[role] not in header:
                sys.exit(f"ERROR: --{role}-col {overrides[role]!r} is not in the header {header}")
            out[role] = overrides[role]
            continue
        hit = next((low[c] for c in cands if c in low), None)
        if hit is None:
            if role in ("bid", "ask") and NOT_QUOTES & set(low):
                sys.exit(f"ERROR: no closing {role} column, but {sorted(NOT_QUOTES & set(low))} found.\n"
                         "  BIDLO/ASKHI (CIZ: DlyLow/DlyHigh) are the day's low/high TRADE prices when the stock "
                         "traded; they are not quotes and would overstate spreads massively.\n"
                         "  Download the closing quotes instead: BID and ASK (legacy) or DlyBid and DlyAsk (CIZ).")
            sys.exit(f"ERROR: cannot find the {role} column in {header}. Use --{role}-col NAME.")
        out[role] = hit
    for role in ("bid", "ask"):
        if out[role].lower() in NOT_QUOTES:
            sys.exit(f"ERROR: {out[role]!r} is a low/high trade price, not a closing quote (see --inspect).")
    return out


def detect_date_format(values: list[str], forced: str | None) -> str:
    vals = [v.strip().split(" ")[0] for v in values if v and v.strip() and v.strip() not in NULLS][:2000]
    if not vals:
        sys.exit("ERROR: no date values in the first rows of the file.")
    if forced:
        return forced

    def share(fmt: str) -> float:
        n = 0
        for v in vals:
            try:
                n += 1920 <= datetime.strptime(v, fmt).year <= 2100
            except ValueError:
                pass
        return n / len(vals)

    scores = {f: share(f) for f in DATE_FORMATS}
    best = max(scores, key=scores.get)
    if scores[best] < 0.9:  # a few unreadable dates are tolerated here and counted in the report
        sys.exit(f"ERROR: unknown date format, e.g. {vals[:3]}. Pass --date-format (strftime codes).")
    if best in ("%m/%d/%Y", "%d/%m/%Y") and scores["%m/%d/%Y"] == scores["%d/%m/%Y"]:
        sys.exit(f"ERROR: dates such as {vals[0]} could be month/day or day/month. Pass --date-format.")
    return best


def fmt_n(x) -> str:
    return "-" if x is None else f"{x:,.0f}"


def pct(a, b) -> str:
    return "-" if not b else f"{100 * a / b:5.1f}%"


# ----------------------------------------------------------------------------------------------- step 1
def step1_daily(con, src: Path, delim: str, cols: dict[str, str], fmt: str, rebuild: bool) -> None:
    stat = src.stat()
    meta = {"version": VERSION, "source": str(src), "size": stat.st_size, "mtime": int(stat.st_mtime),
            "columns": cols, "date_format": fmt, "delim": delim}
    if not rebuild and SPREADS_DAILY.exists() and META.exists():
        try:
            if json.loads(META.read_text()) == meta:
                print(f"[step 1] reusing {SPREADS_DAILY.name} (same source file; --rebuild to force)\n")
                return
        except ValueError:
            pass
    print(f"[step 1] reading {src.name} ({stat.st_size / 1e9:.2f} GB) -> {SPREADS_DAILY.name} ...", flush=True)
    t0 = time.time()
    p, d, b, a = (q(cols[k]) for k in ("permno", "date", "bid", "ask"))
    nulls = "[" + ", ".join("'" + s + "'" for s in NULLS) + "]"
    tmp_out = SPREADS_DAILY.with_suffix(".parquet.partial")
    con.sql(f"""
        COPY (
            WITH raw AS (
                SELECT TRY_CAST(TRY_CAST(trim({p}) AS DOUBLE) AS INTEGER)            AS PERMNO,
                       CAST(try_strptime(split_part(trim({d}), ' ', 1), '{fmt}') AS DATE) AS date,
                       TRY_CAST(trim({b}) AS DOUBLE)                                 AS bid,
                       TRY_CAST(trim({a}) AS DOUBLE)                                 AS ask,
                       ({b} IS NOT NULL AND TRY_CAST(trim({b}) AS DOUBLE) IS NULL)
                    OR ({a} IS NOT NULL AND TRY_CAST(trim({a}) AS DOUBLE) IS NULL)   AS bad_num
                FROM read_csv({paths.sql_path(src)}, header = true, delim = '{delim}', all_varchar = true,
                              nullstr = {nulls}, auto_detect = true))
            SELECT PERMNO, date,
                   avg(bid) AS bid, avg(ask) AS ask,                 -- duplicates merged (see conflict)
                   CAST(count(*) AS INTEGER) AS n_rows,
                   coalesce(min(bid) <> max(bid) OR min(ask) <> max(ask), false) AS conflict,
                   bool_or(bad_num) AS bad_num
            FROM raw GROUP BY PERMNO, date
        ) TO {paths.sql_path(tmp_out)} (FORMAT parquet, COMPRESSION zstd)""")
    tmp_out.replace(SPREADS_DAILY)
    META.write_text(json.dumps(meta, indent=1))
    mb = SPREADS_DAILY.stat().st_size / 1e6
    print(f"[step 1] done in {time.time() - t0:.0f} s; {SPREADS_DAILY.name} = {mb:,.0f} MB "
          "(keep it: re-runs then skip the CSV; delete it to free space)\n")


# ----------------------------------------------------------------------------------------------- step 2
def status_cte(max_rel: float) -> str:
    return f"""
        d AS (SELECT * FROM read_parquet({paths.sql_path(SPREADS_DAILY)})),
        s AS (
            SELECT PERMNO, date, n_rows, conflict, bad_num, bid, ask,
                   CAST(year(date) * 100 + month(date) AS INTEGER) AS YYYYMM,
                   CASE WHEN bid > 0 AND ask > 0 THEN (ask - bid) / ((ask + bid) / 2) END AS rs,
                   CASE WHEN PERMNO IS NULL OR date IS NULL         THEN 'bad_key'
                        WHEN bid IS NULL OR ask IS NULL             THEN 'missing'
                        WHEN bid <= 0 OR ask <= 0                   THEN 'nonpositive'
                        WHEN ask < bid                              THEN 'crossed'
                        WHEN ask = bid                              THEN 'locked'
                        WHEN (ask - bid) / ((ask + bid) / 2) > {max_rel} THEN 'too_wide'
                        ELSE 'ok' END AS status
            FROM d)"""


def step2_monthly(con, max_rel: float) -> None:
    # No median: DuckDB cannot spill holistic aggregates (median/quantile) to disk, and over ~9 million
    # PERMNO-months it exceeded a 3 GB memory limit in testing. Mean and month-end value use little memory.
    print(f"[step 2] daily -> {SPREADS_MONTHLY.name} (max relative spread {max_rel:.0%}) ...", flush=True)
    t0 = time.time()
    tmp_out = SPREADS_MONTHLY.with_suffix(".parquet.partial")
    con.sql(f"""
        COPY (
            WITH {status_cte(max_rel)}
            SELECT PERMNO, YYYYMM,
                   CAST((YYYYMM // 100) * 12 + YYYYMM % 100 - 1 AS INTEGER) AS mi,
                   avg(rs)             FILTER (WHERE status = 'ok') AS spread_mean,
                   arg_max(rs, date)   FILTER (WHERE status = 'ok') AS spread_last,
                   max(date)           FILTER (WHERE status = 'ok') AS last_date,
                   CAST(count(*) FILTER (WHERE status = 'ok') AS SMALLINT) AS n_valid,
                   CAST(count(*) AS SMALLINT)                               AS n_obs
            FROM s WHERE status <> 'bad_key'
            GROUP BY PERMNO, YYYYMM ORDER BY PERMNO, YYYYMM
        ) TO {paths.sql_path(tmp_out)} (FORMAT parquet, COMPRESSION zstd)""")
    tmp_out.replace(SPREADS_MONTHLY)
    print(f"[step 2] done in {time.time() - t0:.0f} s\n")


# ----------------------------------------------------------------------------------------------- report
def report_daily(con, max_rel: float) -> None:
    print("=== 1. Daily file: parsing and duplicates ===")
    r = con.sql(f"""WITH {status_cte(max_rel)}
        SELECT sum(n_rows), count(*), sum(n_rows) FILTER (WHERE status = 'bad_key'),
               count(*) FILTER (WHERE bad_num), count(*) FILTER (WHERE n_rows > 1),
               count(*) FILTER (WHERE conflict), min(date), max(date), count(DISTINCT PERMNO),
               median((bid + ask) / 2) FILTER (WHERE status = 'ok')
        FROM s""").fetchone()
    rows, days, badkey, badnum, dups, conflicts, dmin, dmax, nperm, mid = r
    print(f"CSV rows: {fmt_n(rows)} | PERMNO-days: {fmt_n(days)} | PERMNOs: {fmt_n(nperm)} | {dmin} to {dmax}")
    print(f"rows with unreadable PERMNO or date (dropped): {fmt_n(badkey)}")
    print(f"PERMNO-days with a non-numeric bid/ask (set to missing): {fmt_n(badnum)}")
    print(f"PERMNO-days that appeared more than once: {fmt_n(dups)} (merged); "
          f"of which with different bid/ask values: {fmt_n(conflicts)} (averaged)")
    print(f"median quote midpoint of valid quotes: ${mid:,.2f}" if mid is not None else "no valid quote at all")
    if mid is not None and not 0.5 <= mid <= 500:
        print("  WARNING: implausible price level; check the units of the bid/ask columns.")

    print("\n=== 2. Daily quote validity by decade (share of PERMNO-days) ===")
    df = con.sql(f"""WITH {status_cte(max_rel)}
        SELECT (year(date) // 10) * 10 AS decade, count(*) AS n,
               count(*) FILTER (WHERE status = 'ok')          AS ok,
               count(*) FILTER (WHERE status = 'missing')     AS missing,
               count(*) FILTER (WHERE status = 'nonpositive') AS nonpos,
               count(*) FILTER (WHERE status = 'crossed')     AS crossed,
               count(*) FILTER (WHERE status = 'locked')      AS locked,
               count(*) FILTER (WHERE status = 'too_wide')    AS too_wide
        FROM s WHERE status <> 'bad_key' GROUP BY 1 ORDER BY 1""").df()
    print(f"{'decade':<8}{'PERMNO-days':>14}{'valid':>8}{'missing':>9}{'<=0':>8}{'crossed':>9}"
          f"{'locked':>8}{'>max':>8}")
    for _, x in df.iterrows():
        print(f"{int(x.decade)}s{x.n:>15,.0f}" + "".join(f"{pct(x[c], x.n):>8} " for c in
              ["ok", "missing", "nonpos", "crossed", "locked", "too_wide"]))
    tot = df.sum(numeric_only=True)
    if tot.n and tot.crossed / tot.n > 0.05:
        print("  WARNING: more than 5% crossed quotes (ask < bid): are the bid and ask columns swapped?")


def report_monthly(con) -> None:
    m = paths.sql_path(SPREADS_MONTHLY)
    print("\n=== 3. Monthly file ===")
    r = con.sql(f"""SELECT count(*), count(spread_mean), count(DISTINCT PERMNO), min(YYYYMM), max(YYYYMM)
                    FROM read_parquet({m})""").fetchone()
    print(f"PERMNO-months: {fmt_n(r[0])}, with a spread: {fmt_n(r[1])} ({pct(r[1], r[0]).strip()}); "
          f"PERMNOs: {fmt_n(r[2])}; {r[3]} to {r[4]}")
    nv = con.sql(f"""SELECT quantile_cont(n_valid, [0.1, 0.5, 0.9]) FROM read_parquet({m})
                     WHERE n_valid > 0""").fetchone()[0]
    print(f"valid daily quotes per month (months with >= 1): p10 {nv[0]:.0f}, median {nv[1]:.0f}, p90 {nv[2]:.0f}")
    r = con.sql(f"""
        WITH me AS (SELECT CAST(year(date) * 100 + month(date) AS INTEGER) AS YYYYMM, max(date) AS last_day
                    FROM read_parquet({paths.sql_path(SPREADS_DAILY)}) WHERE date IS NOT NULL GROUP BY 1)
        SELECT avg(CASE WHEN last_date = last_day THEN 1.0 ELSE 0 END), corr(spread_last, spread_mean),
               median(spread_last / spread_mean)
        FROM (SELECT *, CAST(year(last_date) * 100 + month(last_date) AS INTEGER) AS ym FROM read_parquet({m})) x
        JOIN me ON me.YYYYMM = x.ym WHERE spread_mean > 0""").fetchone()
    print(f"spread_last dated on the month's last trading day: {100 * r[0]:.1f}%; "
          f"corr(spread_last, spread_mean) = {r[1]:.3f}; median ratio last/mean = {r[2]:.2f}")

    print("\n=== 4. Median spread_mean across stocks, by year (full quoted spread, %) ===")
    yr = con.sql(f"""SELECT YYYYMM // 100 AS y, median(spread_mean) AS med, count(spread_mean) AS n
                     FROM read_parquet({m}) WHERE spread_mean IS NOT NULL GROUP BY 1 ORDER BY 1""").df()
    cells = [f"{int(x.y)} {100 * x.med:5.2f}" for _, x in yr.iterrows()]
    for i in range(0, len(cells), 6):
        print("   ".join(cells[i:i + 6]))
    late = yr[yr.y >= 2005]
    if len(late) and late.med.median() > 0.02:
        print("  WARNING: median spread above 2% after 2005 is far too wide for closing quotes. "
              "Check that the columns are closing bid/ask and not daily low/high prices.")


def report_coverage(con) -> None:
    print("\n=== 5. Coverage of the stock universe (crsp_panel.parquet) ===")
    if not paths.CRSP_PANEL.exists():
        print(f"skipped: {paths.CRSP_PANEL} not found (run build_panel.py first)")
        return
    pp = paths.sql_path(paths.CRSP_PANEL)
    have = set(con.sql(f"SELECT * FROM read_parquet({pp}) LIMIT 0").columns)
    need = {"PERMNO", "YYYYMM", "me"}
    if not need <= have:
        print(f"skipped: panel lacks {sorted(need - have)}")
        return
    exch = "exch" if "exch" in have else "'all'"
    cond = ["me > 0"] + (["common"] if "common" in have else [])
    if "exch" in have:
        vals = {v for (v,) in con.sql(f"SELECT DISTINCT exch FROM read_parquet({pp})").fetchall()}
        if {"N", "A", "Q"} <= vals:
            cond.append("exch IN ('N', 'A', 'Q')")
    where = " AND ".join(cond)
    print(f"universe: panel rows with {where} (stocks that can be traded at the end of month t)")
    period = " ".join(f"WHEN year BETWEEN {lo} AND {hi} THEN '{lo}-{hi}'" for lo, hi in PERIODS)
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE cov AS
        WITH p AS (SELECT CAST(PERMNO AS BIGINT) AS PERMNO, CAST(YYYYMM AS INTEGER) AS YYYYMM, me,
                          CAST({exch} AS VARCHAR) AS exch FROM read_parquet({pp}) WHERE {where})
        SELECT p.*, p.YYYYMM // 100 AS year, s.spread_mean, s.n_valid
        FROM p LEFT JOIN read_parquet({paths.sql_path(SPREADS_MONTHLY)}) s
               ON s.PERMNO = p.PERMNO AND s.YYYYMM = p.YYYYMM""")

    def table(key: str):
        return con.sql(f"""
            SELECT {key} AS k, exch, count(*) AS stock_months, count(spread_mean) AS with_spread,
                   coalesce(sum(me) FILTER (WHERE spread_mean IS NOT NULL), 0) / sum(me) AS cap_share,
                   median(spread_mean) AS median_spread,
                   sum(spread_mean * me) / sum(me) FILTER (WHERE spread_mean IS NOT NULL) AS vw_spread,
                   median(n_valid) FILTER (WHERE spread_mean IS NOT NULL) AS median_days
            FROM cov GROUP BY 1, 2 ORDER BY 1, 2""").df()

    by_year = table("year")
    paths.ensure_dirs()
    out = paths.TABLES / "spreads_coverage_by_year.csv"
    by_year.rename(columns={"k": "year"}).to_csv(out, index=False)
    by_period = table(f"CASE {period} END")
    by_period["k"] = [k.replace("1900", str(int(by_year.k.min()))).replace("2100", str(int(by_year.k.max())))
                      for k in by_period.k]
    print(f"{'period':<11}{'exch':<6}{'stock-months':>13}{'% stocks':>10}{'% mkt cap':>11}"
          f"{'median spr':>12}{'VW spr':>9}{'med days':>10}")
    for _, x in by_period.iterrows():
        has = x.with_spread > 0
        print(f"{x.k:<11}{x.exch:<6}{x.stock_months:>13,.0f}{pct(x.with_spread, x.stock_months):>10}"
              f"{100 * x.cap_share:>10.1f}%" + (f"{100 * x.median_spread:>11.2f}%{100 * x.vw_spread:>8.2f}%"
                                               f"{x.median_days:>10.0f}" if has else f"{'-':>12}{'-':>9}{'-':>10}"))
    print("(% mkt cap = share of the cap-months covered; VW spr = cap-weighted mean spread of covered stocks;"
          f" med days = median valid quote days per covered month; yearly table: {out.name} in output/tables)")
    yearly = con.sql("""SELECT year, coalesce(sum(me) FILTER (WHERE spread_mean IS NOT NULL), 0) / sum(me) AS c
                        FROM cov GROUP BY 1 HAVING c < 0.9 ORDER BY 1""").fetchall()
    if yearly:
        print(f"WARNING: less than 90% of market cap has a spread in {len(yearly)} years: "
              + ", ".join(f"{y} ({100 * c:.0f}%)" for y, c in yearly)
              + ".\n  The net-of-cost test needs another spread source in those years (context file, §7b).")
    else:
        print("At least 90% of market cap has a spread in every year.")


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--max-rel", type=float, default=0.5)
    ap.add_argument("--memory-limit")
    ap.add_argument("--threads", type=int)
    for role in CANDIDATES:
        ap.add_argument(f"--{role}-col")
    ap.add_argument("--date-format")
    a = ap.parse_args()

    src = find_csv(a.csv)
    lines = read_head(src)
    delim, header, rows = sniff(lines)
    cols = pick_columns(header, {r: getattr(a, f"{r}_col") for r in CANDIDATES})
    fmt = detect_date_format([r[header.index(cols["date"])] for r in rows if len(r) == len(header)],
                             a.date_format)
    if a.inspect:
        print(f"file: {src}  ({src.stat().st_size / 1e9:.2f} GB)   delimiter: {delim!r}")
        print(f"header ({len(header)} columns): {header}")
        for line in lines[1:6]:
            print("  " + line.rstrip())
        print(f"detected columns: {cols}   date format: {fmt}")
        print("Paste this output to the advisor chat before the full run.")
        return

    paths.log_to("build_spreads.txt")
    t0 = time.time()
    print(f"[input  {src}]\n[output {SPREADS_DAILY}]\n[output {SPREADS_MONTHLY}]")
    print(f"columns: {cols} | date format {fmt} | delimiter {delim!r} | duckdb {duckdb.__version__}\n")
    paths.ensure_dirs()
    con = duckdb.connect()
    con.sql(f"SET temp_directory = {paths.sql_path(paths.TMP)}")
    con.sql("SET preserve_insertion_order = false")
    if a.memory_limit:
        con.sql(f"SET memory_limit = '{a.memory_limit}'")
    if a.threads:
        con.sql(f"SET threads = {a.threads}")

    step1_daily(con, src, delim, cols, fmt, a.rebuild)
    step2_monthly(con, a.max_rel)
    report_daily(con, a.max_rel)
    report_monthly(con)
    report_coverage(con)
    print(f"\nTotal time: {time.time() - t0:.0f} s. Send this printout (output/logs/build_spreads.txt) "
          "to the advisor chat.")


if __name__ == "__main__":
    main()