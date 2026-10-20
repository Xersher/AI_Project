"""
validate_vs_french.py: check the clean panel by rebuilding three Ken French series from it.

  1. Market: our value-weighted return minus RF vs French Mkt-RF.
  2. Size deciles: June market cap, NYSE breakpoints, held July-June, value-weighted
     (French "Portfolios_Formed_on_ME", deciles).
  3. Momentum deciles: cumulative return t-12..t-2, NYSE breakpoints, monthly, value-weighted
     (French "10_Portfolios_Prior_12_2").

Usage (from the project folder):   python code/validate_vs_french.py
  Reads data/processed/crsp_panel.parquet. French files are taken from data/raw/factors/ if present,
  otherwise downloaded and saved there. If the download fails, put these files (zip or CSV) in
  data/raw/factors/ by hand: Portfolios_Formed_on_ME, 10_Portfolios_Prior_12_2, F-F_Research_Data_Factors.
  Results are also saved to output/logs/validate_vs_french.txt.
Pass mark: monthly correlations of about 0.99 for the market and size deciles and about 0.98 or
more for momentum deciles, with mean differences of a few basis points.
"""
from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

import paths

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {"me": "Portfolios_Formed_on_ME_CSV.zip", "mom": "10_Portfolios_Prior_12_2_CSV.zip",
         "ff": "F-F_Research_Data_Factors_CSV.zip"}


def french_first_block(raw: bytes) -> pd.DataFrame:
    """First table in a French CSV = monthly (value-weighted) returns. Percent -> decimal."""
    lines = raw.decode("latin-1").splitlines()
    header, rows = None, []
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            if line.strip().startswith(",") and len(cells) >= 2:
                header = ["date"] + cells[1:]
                while header[-1] == "":  # trailing comma(s) in the header line
                    header.pop()
            continue
        if re.fullmatch(r"\d{6}", cells[0] or ""):
            row = cells[: len(header)]
            rows.append(row + [""] * (len(header) - len(row)))  # short rows -> missing values
        elif rows:
            break
    df = pd.DataFrame(rows, columns=header).set_index("date")
    df.index = df.index.astype(int)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.mask(df <= -99.99) / 100


def load_french(key: str, local_dir: Path | None) -> pd.DataFrame:
    name = FILES[key]
    stem = name.lower().replace("_csv.zip", "")
    hits = [] if local_dir is None or not local_dir.exists() else sorted(
        p for p in local_dir.iterdir() if p.is_file() and p.name.lower() in (name.lower(), stem + ".csv"))
    if hits:
        blob = hits[-1].read_bytes()
    else:
        req = urllib.request.Request(FRENCH + name, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            blob = resp.read()
        if local_dir is not None:  # keep a copy so the run is reproducible offline
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / name).write_bytes(blob)
    if blob[:2] != b"PK":  # already unzipped (Safari does this automatically)
        return french_first_block(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return french_first_block(z.read(csv_name))


def deciles(df: pd.DataFrame, what: str) -> pd.DataFrame:
    """The 10 decile columns of a French table, renamed p1..p10.

    Taken by position (the last 10 columns: the whole table for momentum, the columns after the
    30/40/30 and quintile groups for size), not by name, because French renames the middle columns
    between vintages: the 2026 size file has '2-Dec' ... '9-Dec' where older files had 'Dec 2' ... 'Dec 9'.
    """
    out = df.iloc[:, -10:].copy()
    if len(out.columns) != 10 or not (str(out.columns[0]).startswith("Lo") and str(out.columns[-1]).startswith("Hi")):
        raise ValueError(f"unexpected {what} header in the French file: {list(df.columns)}")
    out.columns = [f"p{k}" for k in range(1, 11)]
    return out


def ten_breaks(alias: str) -> str:
    return " + ".join(f"CAST({alias}.x > bp[{k}] AS INTEGER)" for k in range(1, 10))


def compare(ours: pd.DataFrame, theirs: pd.DataFrame, label: str) -> None:
    both = ours.join(theirs, how="inner", lsuffix="_ours", rsuffix="_french").dropna()
    rows = []
    for k in range(1, 11):
        a, b = both[f"p{k}_ours"], both[f"p{k}_french"]
        rows.append([f"D{k}", a.corr(b), 1e4 * a.mean(), 1e4 * b.mean()])
    a = both["p10_ours"] - both["p1_ours"]
    b = both["p10_french"] - both["p1_french"]
    rows.append(["D10-D1", a.corr(b), 1e4 * a.mean(), 1e4 * b.mean()])
    out = pd.DataFrame(rows, columns=["portfolio", "corr", "mean_ours_bp", "mean_french_bp"]).round(3)
    print(f"\n=== {label}: {len(both)} months, {both.index.min()} to {both.index.max()} ===")
    print(out.to_string(index=False))


def main(panel_path: str | None = None, french_dir: str | None = None) -> None:
    paths.log_to("validate_vs_french.txt")
    panel = Path(panel_path).expanduser().resolve() if panel_path else paths.CRSP_PANEL
    local = Path(french_dir).expanduser().resolve() if french_dir else paths.RAW_FACTORS
    print(f"[panel {panel}]\n[French files {local}]\n")
    con = duckdb.connect()
    con.sql(f"""CREATE VIEW p AS SELECT PERMNO, YYYYMM, mi, ret, me, me_lag, exch, exch_lag, common, eligible
                FROM read_parquet({paths.sql_path(panel)})""")
    con.sql("CREATE VIEW u AS SELECT * FROM p WHERE eligible AND ret IS NOT NULL")

    # 1. Market
    mkt = con.sql("SELECT YYYYMM, sum(ret * me_lag) / sum(me_lag) AS vw FROM u GROUP BY 1").df()
    ff = load_french("ff", local)
    m = mkt.set_index("YYYYMM").join(ff, how="inner").dropna()
    ex = m["vw"] - m["RF"]
    print(f"=== Market: {len(m)} months | corr with Mkt-RF {ex.corr(m['Mkt-RF']):.4f} | "
          f"mean ours {1e4 * ex.mean():.1f} bp vs French {1e4 * m['Mkt-RF'].mean():.1f} bp ===")

    # 2. Size deciles: June ME, NYSE breakpoints, held July (t) to June (t+1)
    size = con.sql(f"""
        WITH june AS (SELECT PERMNO, YYYYMM // 100 AS fy, me AS x, exch
                      FROM p WHERE YYYYMM % 100 = 6 AND common AND me > 0),
             bpt AS (SELECT fy, quantile_cont(x, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) AS bp
                     FROM june WHERE exch = 'N' GROUP BY 1),
             asg AS (SELECT j.PERMNO, j.fy, 1 + {ten_breaks('j')} AS dec FROM june j JOIN bpt USING (fy))
        SELECT u.YYYYMM, a.dec, sum(u.ret * u.me_lag) / sum(u.me_lag) AS r
        FROM u JOIN asg a ON u.PERMNO = a.PERMNO
             AND a.fy = CASE WHEN u.YYYYMM % 100 >= 7 THEN u.YYYYMM // 100 ELSE u.YYYYMM // 100 - 1 END
        GROUP BY 1, 2""").df()
    ours_size = size.pivot(index="YYYYMM", columns="dec", values="r").add_prefix("p")
    me = deciles(load_french("me", local), "size")
    compare(ours_size, me, "Size deciles (VW)")

    # 3. Momentum deciles: cumulative return over t-12..t-2, formed at end of t-1
    mom = con.sql(f"""
        WITH s AS (
            SELECT PERMNO, YYYYMM, mi, ret, me_lag, exch_lag, eligible,
                   sum(ln(greatest(1 + ret, 1e-6))) OVER w AS cum, count(ret) OVER w AS n_obs
            FROM p WINDOW w AS (PARTITION BY PERMNO ORDER BY mi RANGE BETWEEN 12 PRECEDING AND 2 PRECEDING)),
             f AS (SELECT * , cum AS x FROM s WHERE eligible AND ret IS NOT NULL AND n_obs = 11 AND me_lag > 0),
             bpt AS (SELECT YYYYMM, quantile_cont(x, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) AS bp
                     FROM f WHERE exch_lag = 'N' GROUP BY 1)
        SELECT f.YYYYMM, 1 + {ten_breaks('f')} AS dec, sum(f.ret * f.me_lag) / sum(f.me_lag) AS r
        FROM f JOIN bpt USING (YYYYMM) GROUP BY 1, 2""").df()
    ours_mom = mom.pivot(index="YYYYMM", columns="dec", values="r").add_prefix("p")
    fm = deciles(load_french("mom", local), "momentum")
    compare(ours_mom, fm, "Momentum deciles (VW)")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        sys.exit("usage: python code/validate_vs_french.py [panel.parquet] [folder_with_french_files]")
    main(*(sys.argv[1:] + [None, None])[:2])