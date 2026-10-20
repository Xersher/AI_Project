"""
get_factors.py: download and align the factors of the seven benchmark models (CAPM, FF3, FF3+Mom,
FF5, FF6 = FF5+Mom, q, q5) into one monthly file.

Usage (from the project folder):   python code/get_factors.py
  Reads the downloads in data/raw/factors/ (any missing file is downloaded and saved there) and writes
  data/processed/factors_monthly.parquet/.csv (YYYYMM, decimal returns) and factors_vintage.txt
  (source files and date: the paper must state the vintage). Printout saved to output/logs/get_factors.txt.

Sources:
  Ken French Data Library: F-F_Research_Data_Factors_CSV.zip, F-F_Research_Data_5_Factors_2x3_CSV.zip,
                           F-F_Momentum_Factor_CSV.zip
  global-q.org (Hou, Mo, Xue & Zhang): q5_factors_monthly_<year>.csv (the newest vintage found)
Columns: rf, mktrf, smb, hml (FF3 file); mktrf5, smb5, hml5, rmw, cma (FF5 file); umd (momentum);
         q_mkt, q_me, q_ia, q_roe, q_eg (q5 file; q_mkt is the market excess return).
"""
from __future__ import annotations

import datetime as dt
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
FRENCH_FILES = {"ff3": "F-F_Research_Data_Factors_CSV.zip",
                "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
                "mom": "F-F_Momentum_Factor_CSV.zip"}
Q_BASE = "https://global-q.org/uploads/1/2/2/6/122679606/"
Q_YEARS = range(dt.date.today().year, 2022, -1)  # try the newest vintage first


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def french_first_block(raw: bytes) -> pd.DataFrame:
    """First table in a French CSV = the monthly series. Percent -> decimal; -99.99 = missing."""
    header, rows = None, []
    for line in raw.decode("latin-1").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            if line.strip().startswith(",") and len(cells) >= 2:
                header = ["date"] + cells[1:]
            continue
        if re.fullmatch(r"\d{6}", cells[0] or ""):
            rows.append(cells[: len(header)])
        elif rows:
            break
    df = pd.DataFrame(rows, columns=header).set_index("date")
    df.index = df.index.astype(int)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.mask(df <= -99.99) / 100


def save_copy(local: Path | None, name: str, blob: bytes) -> None:
    """Keep a copy of every download in data/raw/factors/ so the build is reproducible offline."""
    if local is not None:
        local.mkdir(parents=True, exist_ok=True)
        (local / name).write_bytes(blob)


def find_local(local: Path | None, test) -> Path | None:
    """First file in the folder whose lower-case name passes test()."""
    if local is None or not local.exists():
        return None
    hits = sorted(p for p in local.iterdir() if p.is_file() and test(p.name.lower()))
    return hits[-1] if hits else None


def load_french(key: str, local: Path | None) -> tuple[pd.DataFrame, str]:
    name = FRENCH_FILES[key]
    stem = name.lower().replace("_csv.zip", "")
    # Accept the zip, or the CSV inside it (Safari unzips downloads automatically).
    hit = find_local(local, lambda n: n == name.lower() or n == stem + ".csv")
    if hit is not None:
        blob, origin = hit.read_bytes(), f"local file {hit.name}"
    else:
        blob, origin = fetch(FRENCH + name), FRENCH + name
        save_copy(local, name, blob)
    if blob[:2] == b"PK":  # zip archive: take the CSV inside
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            blob = z.read(next(n for n in z.namelist() if n.lower().endswith(".csv")))
    # U.S. research factors are built from CRSP; international files (e.g. "Developed") from Bloomberg.
    first_line = blob.decode("latin-1").lstrip().splitlines()[0]
    if "CRSP" not in first_line:
        raise ValueError(f"{name}: first line is '{first_line}'. This is not a U.S. (CRSP-based) file; "
                         "download the U.S. Research Returns Data version.")
    return french_first_block(blob), origin


def load_q5(local: Path | None) -> tuple[pd.DataFrame, str]:
    candidates = [f"q5_factors_monthly_{y}.csv" for y in Q_YEARS]
    raw, origin = None, None
    # Accept any monthly q5 CSV in the folder, whatever its vintage year in the name.
    hit = find_local(local, lambda n: "q5" in n and "monthly" in n and n.endswith(".csv"))
    if hit is not None:
        raw, origin = hit.read_bytes(), f"local file {hit.name}"
    if raw is None:
        for name in candidates:
            try:
                raw, origin = fetch(Q_BASE + name), Q_BASE + name
                save_copy(local, name, raw)
                break
            except Exception:
                continue
    if raw is None:
        raise RuntimeError("q5 file not found online: download the monthly q5 file from "
                           "global-q.org/factors.html and save it in data/raw/factors/.")
    q = pd.read_csv(io.BytesIO(raw))
    q.columns = [c.strip().lower() for c in q.columns]
    q.index = (q["year"] * 100 + q["month"]).astype(int)
    names = {"r_mkt": "q_mkt", "r_me": "q_me", "r_ia": "q_ia", "r_roe": "q_roe", "r_eg": "q_eg"}
    missing = [c for c in names if c not in q.columns]
    if missing:
        raise RuntimeError(f"unexpected q5 columns {list(q.columns)}; missing {missing}")
    return q[list(names)].rename(columns=names) / 100, origin


def main(out_dir: str | None = None, local_dir: str | None = None) -> None:
    paths.log_to("get_factors.txt")
    out = Path(out_dir).expanduser().resolve() if out_dir else paths.PROCESSED
    out.mkdir(parents=True, exist_ok=True)
    local = Path(local_dir).expanduser().resolve() if local_dir else paths.RAW_FACTORS
    print(f"[downloads {local}]\n[output {out}]\n")

    ff3, o3 = load_french("ff3", local)
    ff5, o5 = load_french("ff5", local)
    mom, om = load_french("mom", local)
    q5, oq = load_q5(local)

    ff3 = ff3.rename(columns={"Mkt-RF": "mktrf", "SMB": "smb", "HML": "hml", "RF": "rf"})
    ff5 = ff5.rename(columns={"Mkt-RF": "mktrf5", "SMB": "smb5", "HML": "hml5", "RMW": "rmw",
                              "CMA": "cma", "RF": "rf5"})
    mom = mom.rename(columns={mom.columns[0]: "umd"})
    f = ff3.join([ff5, mom, q5], how="outer").sort_index()
    f.index.name = "YYYYMM"

    # Consistency checks between files (printed so the team can cite them).
    # Each model uses the factors of its own file (FF3-file Mkt-RF for CAPM/FF3/FF3+Mom, FF5-file
    # Mkt-RF for FF5/FF6), so small differences between files are harmless; they are reported here.
    both = f.dropna(subset=["mktrf", "mktrf5"])
    for a, b, label in [("mktrf", "mktrf5", "Mkt-RF"), ("hml", "hml5", "HML"), ("rf", "rf5", "RF")]:
        d = (both[a] - both[b]).abs()
        print(f"{label}, FF3 vs FF5 file: {int((d > 1e-9).sum())} of {len(d)} months differ, "
              f"max difference {100 * d.max():.2f} percentage points")
    f = f.drop(columns="rf5")

    f.reset_index().to_csv(out / "factors_monthly.csv", index=False)
    duckdb.from_df(f.reset_index()).write_parquet(str(out / "factors_monthly.parquet"))
    with open(out / "factors_vintage.txt", "w") as fh:
        fh.write(f"downloaded {dt.datetime.now():%Y-%m-%d %H:%M}\n")
        for o in (o3, o5, om, oq):
            fh.write(o + "\n")

    print("\nFirst and last month with data, by column:")
    span = pd.DataFrame({c: [f[c].first_valid_index(), f[c].last_valid_index()] for c in f.columns},
                        index=["first", "last"]).T
    print(span.to_string())
    print(f"\n[wrote factors_monthly.parquet/.csv and factors_vintage.txt to {out}]")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        sys.exit("usage: python code/get_factors.py [output_folder] [folder_with_downloaded_files]")
    main(*(sys.argv[1:] + [None, None])[:2])