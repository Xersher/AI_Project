"""
paths.py: every folder and file the project uses, defined once.

All scripts import this file, so nobody has to type a path. It finds the project folder by itself
(the folder that contains code/), so it works unchanged on every teammate's computer as long as
everyone keeps the same layout:

    applied_investments/
    ├── code/               this file and all scripts
    ├── data/raw/crsp/      the CRSP CSV (any file name, exactly one .csv)
    ├── data/raw/factors/   the Ken French and q5 downloads
    ├── data/processed/     files the scripts create
    └── output/             tables/, figures/, logs/ (every script saves a copy of its printout here)

If your data folder is somewhere else (e.g. a shared drive), create a file code/paths_local.py
containing one line, and do not share that file (each teammate has their own):
    DATA = "/Users/yourname/path/to/data"

Check your setup:   python code/paths.py
"""
from __future__ import annotations

import atexit
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent

try:  # optional personal override, see above
    from paths_local import DATA as _LOCAL_DATA
    DATA = Path(_LOCAL_DATA).expanduser().resolve()
except ImportError:
    DATA = ROOT / "data"

# Folders
RAW = DATA / "raw"
RAW_CRSP = RAW / "crsp"
RAW_FACTORS = RAW / "factors"
PROCESSED = DATA / "processed"
TMP = PROCESSED / "duckdb_tmp"  # DuckDB spill space for large queries
OUTPUT = ROOT / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
LOGS = OUTPUT / "logs"

# Files created by the data scripts
CRSP_MONTHLY = PROCESSED / "crsp_monthly.parquet"  # data_audit.py
CRSP_PANEL = PROCESSED / "crsp_panel.parquet"      # build_panel.py
FACTORS = PROCESSED / "factors_monthly.parquet"    # get_factors.py


def crsp_csv() -> Path:
    """The raw CRSP CSV: the single .csv file in data/raw/crsp/."""
    files = sorted(RAW_CRSP.glob("*.csv")) if RAW_CRSP.exists() else []
    if len(files) != 1:
        msg = f"expected exactly one .csv in {RAW_CRSP}, found {len(files)}"
        stray = sorted(RAW.glob("*.csv")) if RAW.exists() else []
        if not files and stray:  # the usual mistake: CSV put in data/raw/ instead of data/raw/crsp/
            msg += f"; {', '.join(p.name for p in stray)} is directly in {RAW}: move it into {RAW_CRSP}"
        raise FileNotFoundError(msg)
    return files[0]


def sql_path(p: Path | str) -> str:
    """A path as a quoted DuckDB string literal (safe with spaces and apostrophes)."""
    return "'" + str(p).replace("'", "''") + "'"


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet (never touches data/raw)."""
    for d in (PROCESSED, TMP, OUTPUT, TABLES, FIGURES, LOGS):
        d.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            if not getattr(st, "closed", False):
                st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            if not getattr(st, "closed", False):
                st.flush()


def log_to(name: str) -> Path:
    """From now on, also save everything the script prints to output/logs/<name>."""
    ensure_dirs()
    path = LOGS / name
    fh = open(path, "w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = _Tee(original, fh)

    def _close() -> None:  # restore the screen output first, then close the log file
        sys.stdout.flush()
        sys.stdout = original
        fh.close()

    atexit.register(_close)
    return path


if __name__ == "__main__":
    print(f"project folder : {ROOT}")
    print(f"data folder    : {DATA}" + ("   (from paths_local.py)" if DATA != ROOT / "data" else ""))
    for label, p in [("raw CRSP", RAW_CRSP), ("raw factors", RAW_FACTORS), ("processed", PROCESSED),
                     ("output", OUTPUT), ("crsp_monthly", CRSP_MONTHLY), ("crsp_panel", CRSP_PANEL),
                     ("factors", FACTORS)]:
        print(f"{label:<15}: {'OK     ' if p.exists() else 'MISSING'} {p}")
    try:
        print(f"{'CRSP CSV':<15}: OK      {crsp_csv()}")
    except FileNotFoundError as e:
        print(f"{'CRSP CSV':<15}: MISSING {e}")
