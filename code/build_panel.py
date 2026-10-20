"""
build_panel.py: turn crsp_monthly.parquet (made by data_audit.py) into one clean monthly panel.

Usage (from the project folder):   python code/build_panel.py
  Reads data/processed/crsp_monthly.parquet, writes data/processed/crsp_panel.parquet (one row per
  PERMNO-month); the summary is also saved to output/logs/build_panel.txt. Paths come from paths.py.

Every rule below is a project decision recorded in context_ai_project.md (sections 7 and 13):
  1. One row per PERMNO-month. The raw export repeats a month when several distribution or
     share records fall in it; returns are identical across the copies (audit, 16 Oct).
  2. Eligibility for the month-t return is decided on the month t-1 row (common stock, NYSE/AMEX/
     Nasdaq, positive market cap), so delisting-month returns are never dropped.
  3. Returns CRSP flags as missing (MthRetFlg NT = not tracked, GP = price gap > 10 days) stay missing.
  4. Delisting returns CRSP excluded (MthDelFlg M = missing, G = gap) for exchange-dropped
     delistings (DelActionType GDR, the legacy 500-series) get a Shumway-type return:
     -30% if the stock was on NYSE/AMEX at t-1, -55% if on Nasdaq (Shumway 1997; Shumway & Warther 1999).
Units: returns are decimals; market caps in $ thousands; turnover = shares traded / shares outstanding.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb

import paths

COMMON = """
    coalesce(ShareType = 'NS', false) AND coalesce(SecurityType = 'EQTY', false)
    AND coalesce(SecuritySubType = 'COM', false) AND coalesce(USIncFlg = 'Y', false)
    AND coalesce(IssuerType IN ('ACOR', 'CORP'), false)
    AND coalesce(PrimaryExch IN ('N', 'A', 'Q'), false)
    AND coalesce(ConditionalType IN ('RW', 'NW'), false)
    AND coalesce(TradingStatusFlg = 'A', false)
"""
SHUMWAY_NYSE_AMEX, SHUMWAY_NASDAQ = -0.30, -0.55


def main(src_path: str | None = None) -> None:
    paths.log_to("build_panel.txt")
    src = Path(src_path).expanduser().resolve() if src_path else paths.CRSP_MONTHLY
    out = paths.CRSP_PANEL
    print(f"[input {src}]\n[output {out}]\n")
    t0 = time.time()
    con = duckdb.connect()
    con.sql(f"SET temp_directory = {paths.sql_path(paths.TMP)}")
    con.sql(f"CREATE VIEW raw AS SELECT * FROM read_parquet({paths.sql_path(src)})")

    # Consistency of the duplicated months (decides whether any_value is safe).
    chk = con.sql("""
        SELECT count(*) AS dup_groups,
               sum(CASE WHEN n_ret > 1 THEN 1 ELSE 0 END) AS diff_return,
               sum(CASE WHEN n_cap > 1 THEN 1 ELSE 0 END) AS diff_mthcap,
               sum(CASE WHEN n_shr > 1 THEN 1 ELSE 0 END) AS diff_shrout
        FROM (SELECT PERMNO, YYYYMM, count(*) AS n, count(DISTINCT MthRet) AS n_ret,
                     count(DISTINCT MthCap) AS n_cap, count(DISTINCT ShrOut) AS n_shr
              FROM raw GROUP BY ALL HAVING n > 1)""").fetchone()
    print(f"duplicate PERMNO-months: {chk[0]:,} | with different MthRet: {chk[1]:,} | "
          f"different MthCap: {chk[2]:,} | different ShrOut: {chk[3]:,}")
    if chk[1] or chk[2]:
        print("WARNING: duplicates disagree on returns or market cap; send this line to the advisor.")

    # Rule 1: one row per PERMNO-month. Shares are implied from CRSP's month-end cap (MthCap / MthPrc),
    # so a within-month change in shares outstanding cannot pick the wrong record.
    con.sql(f"""
        CREATE TABLE d AS
        SELECT PERMNO, YYYYMM, (YYYYMM // 100) * 12 + (YYYYMM % 100) - 1 AS mi,
               any_value(PERMCO) AS PERMCO, any_value(PrimaryExch) AS exch,
               any_value(SICCD) AS sic, any_value(MthRet) AS MthRet, any_value(MthRetx) AS MthRetx,
               any_value(MthRetFlg) AS MthRetFlg, any_value(MthDelFlg) AS MthDelFlg,
               any_value(MthPrc) AS prc, any_value(MthCap) AS me, any_value(MthPrevCap) AS prevcap,
               any_value(MthVol) AS vol,
               bool_and({COMMON}) AS common
        FROM raw GROUP BY PERMNO, YYYYMM""")

    # Delisting action and reason sit on a stock's earlier rows, not on the delisting row.
    con.sql("""
        CREATE TABLE dl AS
        SELECT PERMNO,
               arg_max(DelActionType, YYYYMM) FILTER (WHERE DelActionType IS NOT NULL) AS del_action,
               arg_max(DelReasonType, YYYYMM) FILTER (WHERE DelReasonType IS NOT NULL) AS del_reason
        FROM raw GROUP BY 1""")

    con.sql(f"""
        CREATE TABLE p AS
        WITH lagged AS (
            SELECT d.*,
                   lag(mi) OVER w AS mi_lag, lag(common) OVER w AS common_lag,
                   lag(me) OVER w AS me_prev_row, lag(exch) OVER w AS exch_lag, lag(sic) OVER w AS sic_lag
            FROM d WINDOW w AS (PARTITION BY PERMNO ORDER BY mi)),
        r AS (
            SELECT l.*, dl.del_action, dl.del_reason,
                   CASE WHEN MthRetFlg IN ('NT', 'GP') THEN NULL ELSE MthRet END AS ret_crsp,
                   CASE WHEN MthRetFlg IN ('NT', 'GP') THEN NULL ELSE MthRetx END AS retx,
                   (MthDelFlg IN ('M', 'G') AND dl.del_action = 'GDR') AS shumway,
                   (mi_lag = mi - 1) AS contiguous
            FROM lagged l LEFT JOIN dl USING (PERMNO))
        SELECT PERMNO, PERMCO, YYYYMM, mi,
               CASE WHEN shumway THEN (1 + coalesce(ret_crsp, 0))
                                      * (1 + CASE WHEN exch_lag IN ('N', 'A') THEN {SHUMWAY_NYSE_AMEX}
                                                  ELSE {SHUMWAY_NASDAQ} END) - 1
                    ELSE ret_crsp END AS ret,
               ret_crsp, retx,
               CASE WHEN contiguous THEN me_prev_row ELSE NULL END AS me_lag,
               me, prc, me / nullif(prc, 0) AS shrout_k,
               vol / nullif(1000 * me / nullif(prc, 0), 0) AS turnover,
               coalesce(contiguous AND common_lag AND me_prev_row > 0, false) AS eligible,
               common, exch, exch_lag, sic, sic_lag,
               MthRetFlg, MthDelFlg, del_action, del_reason, coalesce(shumway, false) AS shumway
        FROM r""")
    con.sql(f"COPY p TO {paths.sql_path(out)} (FORMAT parquet, COMPRESSION zstd)")

    print(f"\n[wrote {out.name} in {time.time() - t0:.0f}s]")
    print(con.sql("""
        SELECT count(*) AS rows, count(DISTINCT PERMNO) AS permnos, min(YYYYMM) AS first_month,
               max(YYYYMM) AS last_month,
               sum(CASE WHEN eligible THEN 1 ELSE 0 END) AS eligible_rows,
               sum(CASE WHEN eligible AND ret IS NULL THEN 1 ELSE 0 END) AS eligible_missing_ret
        FROM p""").df().to_string(index=False))
    print("\nDelisting months kept in the eligible universe, by CRSP delisting flag and action:")
    print(con.sql("""
        SELECT MthDelFlg, del_action, count(*) AS n,
               sum(CASE WHEN shumway THEN 1 ELSE 0 END) AS shumway_imputed,
               round(median(ret_crsp), 4) AS median_ret_crsp, round(median(ret), 4) AS median_ret_used
        FROM p WHERE eligible AND MthDelFlg <> 'N' GROUP BY ALL ORDER BY n DESC LIMIT 12""").df().to_string(index=False))
    print("\nShumway imputations by exchange at t-1:")
    print(con.sql("""
        SELECT exch_lag, count(*) AS n FROM p WHERE eligible AND shumway GROUP BY 1 ORDER BY n DESC
    """).df().to_string(index=False))
    print("\nEligible stocks per month (Dec of selected years) and median turnover:")
    print(con.sql("""
        SELECT YYYYMM // 100 AS year, count(*) AS stocks,
               round(median(turnover), 4) AS median_monthly_turnover
        FROM p WHERE eligible AND YYYYMM % 100 = 12 AND YYYYMM // 100 IN (1968, 1973, 1990, 2000, 2010, 2025)
        GROUP BY 1 ORDER BY 1""").df().to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit("usage: python code/build_panel.py [path/to/other_crsp_monthly.parquet]")
    main(sys.argv[1] if len(sys.argv) == 2 else None)