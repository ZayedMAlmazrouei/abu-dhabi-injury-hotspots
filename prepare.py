"""
Stack + clean the Abu Dhabi Police traffic-accident XLSX files (2022-2025).

Usage:
    python prepare.py <folder_with_xlsx_files>

Output (in ./out/):
    accidents_raw.parquet     all sheets of all files, stacked, headers normalised
    accidents_clean.parquet   dates parsed, labels mapped, duplicates flagged
    quality_report.md         missing / duplicate / label-variant summary
"""
import re, sys, glob, os
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = "out"; os.makedirs(OUT, exist_ok=True)

# Sheet name -> severity class (sheet names are Arabic/English; match loosely)
SHEET_CLASS = [  # order matters: no-injury before injury
    (r"unknown|مجهول|hit",            "hit_and_run"),
    (r"no.?injur|بدون|بسيط|damage",   "no_injury"),
    (r"injur|اصاب|إصاب",            "injury"),
]

def classify_sheet(name: str) -> str:
    n = name.lower()
    for pat, label in SHEET_CLASS:
        if re.search(pat, n):
            return label
    return "unknown_sheet:" + name

# Bilingual headers with newlines -> short English keys
HEADER_MAP = {
    r"rep.?date":        "date",
    r"rep.?year":        "year",
    r"rep.?time":        "time",
    r"rep.?kind":        "kind",
    r"rep.?type":        "collision_type",
    r"city":             "city",
    r"street":           "street",
    r"reason":           "reason",
    r"surface":          "surface",
    r"weather":          "weather",
    r"no.?of.?accid":    "n_accidents",
    r"no.?of.?injur":    "n_injured",
}

def norm_header(h: str) -> str:
    h = str(h).replace("\n", " ").strip().lower()
    for pat, key in HEADER_MAP.items():
        if re.search(pat, h):
            return key
    return re.sub(r"\W+", "_", h).strip("_") or "unnamed"

def region_from_filename(f: str) -> str:
    f = f.lower()
    if "ain" in f:    return "al_ain"
    if "dhafra" in f or "dafrah" in f: return "al_dhafra"
    return "abu_dhabi"

frames = []
for path in sorted(glob.glob(os.path.join(SRC, "*.xlsx"))):
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df = df.dropna(axis=1, how="all")           # phantom empty columns
        df.columns = [norm_header(c) for c in df.columns]
        df["severity"] = classify_sheet(sheet)
        df["region"] = region_from_filename(os.path.basename(path))
        df["source_file"] = os.path.basename(path)
        df["source_sheet"] = sheet
        frames.append(df)
        print(f"{os.path.basename(path)} / {sheet}: {len(df)} rows -> {df['severity'].iloc[0]}")

raw = pd.concat(frames, ignore_index=True)
for c in raw.select_dtypes("object"):          # mixed types -> string for parquet
    raw[c] = raw[c].astype(str).where(raw[c].notna(), None)
raw.to_parquet(f"{OUT}/accidents_raw.parquet", index=False)

# ---------- cleaning ----------
df = raw.copy()

# Excel serial dates/times -> datetime
if "date" in df:
    df["date"] = pd.to_datetime(
        pd.to_numeric(df["date"], errors="coerce"), unit="D", origin="1899-12-30", errors="coerce"
    ).fillna(pd.to_datetime(df["date"], errors="coerce"))
if "time" in df:
    t = pd.to_numeric(df["time"], errors="coerce")
    hour_num = ((t % 1) * 24).round()
    hour_dt = pd.to_datetime(df["time"], errors="coerce").dt.hour
    df["hour"] = hour_num.fillna(hour_dt).astype("Int64")

df["dow"] = df["date"].dt.dayofweek if "date" in df else pd.NA
df["month"] = df["date"].dt.month if "date" in df else pd.NA

# trim whitespace in all text columns
for c in df.select_dtypes("object"):
    df[c] = df[c].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

# exact duplicates (same file+sheet, identical row) -> flag, don't drop yet
df["is_duplicate"] = df.duplicated(keep="first")

df.to_parquet(f"{OUT}/accidents_clean.parquet", index=False)

# ---------- quality report ----------
lines = [f"# Quality report\n", f"Rows: {len(df)}  |  duplicates flagged: {int(df.is_duplicate.sum())}\n"]
lines.append("## Rows per file/sheet/severity\n")
lines.append(df.groupby(["source_file", "severity"]).size().to_string() + "\n")
lines.append("## Missing per column (%)\n")
lines.append((df.isna().mean() * 100).round(1).to_string() + "\n")
for col in ["collision_type", "reason", "surface", "weather", "city", "kind"]:
    if col in df:
        vc = df[col].value_counts(dropna=False)
        lines.append(f"## {col}: {df[col].nunique()} distinct values\n")
        lines.append(vc.head(40).to_string() + "\n")
open(f"{OUT}/quality_report.md", "w", encoding="utf-8").write("\n".join(lines))
print("\nWrote out/accidents_clean.parquet and out/quality_report.md")
