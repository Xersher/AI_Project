"""
data_audit.py (v2): audit the team's CRSP monthly file (CIZ format) without loading it into memory.

Works on files of any size: DuckDB streams the CSV from disk.

Usage (from the project folder):   python code/data_audit.py
  Finds the CSV in data/raw/crsp/ by itself (see paths.py). A different CSV can be passed as argument.

What it does:
  1. Converts the CSV once into a compact Parquet file, data/processed/crsp_monthly.parquet
     (~30 columns). Re-runs reuse it; delete it to force a rebuild.
  2. Prints a short report, also saved to output/logs/data_audit.txt.

Expected run time on a laptop: under a few minutes for the first run (conversion), seconds afterwards.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb

import paths

KEEP = {  # column -> SQL type (everything is read as text, then cast safely)
    "PERMNO": "INTEGER", "PERMCO": "INTEGER", "YYYYMM": "INTEGER",
    "PrimaryExch": "VARCHAR", "ConditionalType": "VARCHAR", "TradingStatusFlg": "VARCHAR",
    "USIncFlg": "VARCHAR", "IssuerType": "VARCHAR", "SecurityType": "VARCHAR",
    "SecuritySubType": "VARCHAR", "ShareType": "VARCHAR",
    "MthRet": "DOUBLE", "MthRetx": "DOUBLE", "MthRetFlg": "VARCHAR", "MthDelFlg": "VARCHAR",
    "MthPrc": "DOUBLE", "MthPrcFlg": "VARCHAR", "MthCap": "DOUBLE", "MthPrevCap": "DOUBLE",
    "ShrOut": "DOUBLE", "MthVol": "DOUBLE",
    "DelActionType": "VARCHAR", "DelReasonType": "VARCHAR", "DelPaymentType": "VARCHAR",
    "SICCD": "INTEGER", "NAICS": "VARCHAR", "DisType": "VARCHAR", "DisDivAmt": "DOUBLE",
    "DisFacShr": "DOUBLE", "vwretd": "DOUBLE",
}

# CIZ equivalents of legacy shrcd in (10, 11) and exchcd in (1, 2, 3).
# Verify against the CRSP SIZ-to-CIZ cross-reference guide before final use.
COMMON = """
    coalesce(ShareType = 'NS', false) AND coalesce(SecurityType = 'EQTY', false)
    AND coalesce(SecuritySubType = 'COM', false) AND coalesce(USIncFlg = 'Y', false)
    AND coalesce(IssuerType IN ('ACOR', 'CORP'), false)
    AND coalesce(PrimaryExch IN ('N', 'A', 'Q'), false)
    AND coalesce(ConditionalType IN ('RW', 'NW'), false)
    AND coalesce(TradingStatusFlg = 'A', false)
"""


def show(con, title, sql):
    print(f"\n=== {title} ===")
    print(con.sql(sql).df().to_string(index=False))


def main(csv_path: str | None = None) -> None:
    paths.log_to("data_audit.txt")
    src = Path(csv_path).expanduser().resolve() if csv_path else paths.crsp_csv()
    pq = paths.CRSP_MONTHLY
    print(f"[input {src}]\n[output {pq}]\n")
    con = duckdb.connect()
    con.sql(f"SET temp_directory = {paths.sql_path(paths.TMP)}")
    con.sql("SET preserve_insertion_order = false")

    # The CRSP export writes missing values as the text "N/A"; treat it (and empty fields) as missing.
    reader = f"read_csv({paths.sql_path(src)}, header = true, all_varchar = true, nullstr = ['N/A', ''])"
    header = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM {reader}").fetchall()]
    print(f"=== 0. Header: {len(header)} columns ===")
    print(", ".join(header))
    hits = [c for c in header if any(k in c.lower() for k in ("bid", "ask", "spread", "sprd"))]
    print(f"spread-like columns: {hits or 'NONE'}")

    if not pq.exists():
        t0 = time.time()
        cols = ",\n".join(f'TRY_CAST("{c}" AS {t}) AS "{c}"' for c, t in KEEP.items() if c in header)
        con.sql(f"COPY (SELECT {cols} FROM {reader}) TO {paths.sql_path(pq)} (FORMAT parquet, COMPRESSION zstd)")
        print(f"\n[converted to {pq.name} in {time.time() - t0:.0f}s]")
    con.sql(f"CREATE VIEW m AS SELECT * FROM read_parquet({paths.sql_path(pq)})")
    con.sql(f"CREATE VIEW u AS SELECT * FROM m WHERE {COMMON}")

    show(con, "1. Coverage", """
        SELECT count(*) AS rows, count(DISTINCT PERMNO) AS permnos,
               min(YYYYMM) AS first_month, max(YYYYMM) AS last_month,
               count(DISTINCT YYYYMM) AS months FROM m""")

    show(con, "2. Duplicate PERMNO-month rows", """
        WITH g AS (SELECT PERMNO, YYYYMM, count(*) AS n, count(DISTINCT MthRet) AS n_ret
                   FROM m GROUP BY ALL HAVING n > 1)
        SELECT count(*) AS dup_groups, coalesce(sum(n), 0) AS rows_involved,
               coalesce(sum(CASE WHEN n_ret > 1 THEN 1 ELSE 0 END), 0) AS groups_with_different_returns
        FROM g""")

    print("\n=== 3. Filter-field values (top 10 each) ===")
    for c in ["ShareType", "SecurityType", "SecuritySubType", "USIncFlg", "IssuerType",
              "PrimaryExch", "ConditionalType", "TradingStatusFlg", "MthRetFlg", "MthDelFlg"]:
        rows = con.sql(f'SELECT "{c}", count(*) n FROM m GROUP BY 1 ORDER BY n DESC LIMIT 10').fetchall()
        print(c, {r[0]: r[1] for r in rows})

    show(con, "4a. Common-stock universe (row-level filters)", """
        WITH pm AS (SELECT YYYYMM, count(DISTINCT PERMNO) AS n FROM u GROUP BY 1)
        SELECT (SELECT count(DISTINCT PERMNO) FROM u) AS permnos,
               min(YYYYMM) AS first_month, min(n) AS min_per_month,
               CAST(median(n) AS INTEGER) AS median_per_month, max(n) AS max_per_month FROM pm""")
    show(con, "4b. Distinct common stocks by exchange (N = NYSE, A = AMEX, Q = Nasdaq), December of selected years", """
        SELECT YYYYMM // 100 AS year,
               count(DISTINCT PERMNO) FILTER (WHERE PrimaryExch = 'N') AS nyse,
               count(DISTINCT PERMNO) FILTER (WHERE PrimaryExch = 'A') AS amex,
               count(DISTINCT PERMNO) FILTER (WHERE PrimaryExch = 'Q') AS nasdaq
        FROM u WHERE YYYYMM IN (192612, 194012, 196212, 196312, 197312, 199012, 200012, 201012, 202412, 202512)
        GROUP BY 1 ORDER BY 1""")

    con.sql("""CREATE VIEW reason AS
        SELECT PERMNO,
               arg_max(DelReasonType, YYYYMM) FILTER (WHERE DelReasonType IS NOT NULL) AS reason,
               arg_max(DelActionType, YYYYMM) FILTER (WHERE DelActionType IS NOT NULL) AS action,
               arg_max(PrimaryExch, YYYYMM) FILTER (WHERE PrimaryExch IN ('N', 'A', 'Q')) AS last_exch
        FROM m GROUP BY 1""")
    con.sql("CREATE VIEW de AS SELECT m.*, r.reason, r.action, r.last_exch FROM m JOIN reason r USING (PERMNO) "
            "WHERE MthRetFlg = 'DE'")
    show(con, "5a. Delisting months (MthRetFlg = 'DE')", f"""
        SELECT count(*) AS delisting_rows,
               sum(CASE WHEN NOT ({COMMON}) THEN 1 ELSE 0 END) AS dropped_by_row_filters,
               sum(CASE WHEN MthRet IS NULL THEN 1 ELSE 0 END) AS missing_return,
               sum(CASE WHEN MthRet = 0 THEN 1 ELSE 0 END) AS zero_return,
               round(avg(MthRet), 4) AS mean_return, round(median(MthRet), 4) AS median_return
        FROM de""")
    show(con, "5b. Delisting months by reason (top 15)", """
        SELECT reason, action, count(*) AS n,
               sum(CASE WHEN MthRet = 0 THEN 1 ELSE 0 END) AS zero_ret,
               sum(CASE WHEN MthRet IS NULL THEN 1 ELSE 0 END) AS missing_ret,
               round(median(MthRet), 4) AS median_ret
        FROM de GROUP BY ALL ORDER BY n DESC LIMIT 15""")
    show(con, "5c. Delisting months by last exchange", """
        SELECT last_exch, count(*) AS n, round(median(MthRet), 4) AS median_ret,
               sum(CASE WHEN MthRet = 0 THEN 1 ELSE 0 END) AS zero_ret
        FROM de GROUP BY 1 ORDER BY n DESC""")

    show(con, "6. Missing returns in the universe, by decade", """
        SELECT (YYYYMM // 1000) * 10 AS decade, count(*) AS rows,
               round(avg(CASE WHEN MthRet IS NULL THEN 1.0 ELSE 0.0 END), 4) AS share_missing
        FROM u GROUP BY 1 ORDER BY 1""")

    show(con, "7. Units", """
        SELECT round(median(MthCap / (MthPrc * ShrOut)), 4) AS cap_over_prc_x_shrout,
               sum(CASE WHEN MthPrc < 0 THEN 1 ELSE 0 END) AS negative_prices,
               round(median(MthPrc), 2) AS median_price
        FROM m WHERE MthPrc IS NOT NULL AND ShrOut > 0 AND MthCap IS NOT NULL""")

    show(con, "8. Our VW return vs CRSP vwretd", """
        WITH one AS (SELECT PERMNO, YYYYMM, any_value(MthRet) AS r, any_value(MthPrevCap) AS w
                     FROM u WHERE MthRet IS NOT NULL AND MthPrevCap > 0 GROUP BY 1, 2),
             vw AS (SELECT YYYYMM, sum(r * w) / sum(w) AS ours FROM one GROUP BY 1),
             idx AS (SELECT YYYYMM, any_value(vwretd) AS vwretd FROM m GROUP BY 1)
        SELECT count(*) AS months, round(corr(ours, vwretd), 4) AS correlation,
               round(avg(ours - vwretd), 5) AS mean_diff
        FROM vw JOIN idx USING (YYYYMM) WHERE vwretd IS NOT NULL""")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit("usage: python code/data_audit.py [path/to/other.csv]")
    main(sys.argv[1] if len(sys.argv) == 2 else None)