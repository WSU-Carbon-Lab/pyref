"""
Run the NEXAFS beamtime pipeline: discover files, build sample/izero/nexafs catalogs, create DB, ingest.
"""
from __future__ import annotations

import argparse
import io
import re
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection, connect
from typing import Any

import pandas as pd

SCHEMA_SQL = """
DROP TABLE IF EXISTS nexafs;
DROP TABLE IF EXISTS izero;
DROP TABLE IF EXISTS sample;

CREATE TABLE sample (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    tag TEXT NOT NULL,
    version INTEGER NOT NULL,
    beamtime DATE NOT NULL,
    chemical_formula TEXT,
    UNIQUE(name, tag, version, beamtime)
);

CREATE TABLE izero (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    beamline_energy REAL,
    time_stamp TEXT NOT NULL,
    photodiode REAL,
    ai_3_izero REAL
);

CREATE TABLE nexafs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    sample_id INTEGER NOT NULL,
    time_stamp TEXT NOT NULL,
    sample_theta REAL,
    beamline_energy REAL,
    tey_signal REAL,
    ai_3_izero REAL,
    photodiode REAL,
    izero_before_scan_id INTEGER,
    izero_after_scan_id INTEGER,
    FOREIGN KEY (sample_id) REFERENCES sample(id)
);
"""

SAMPLE_COLS = ["name", "tag", "version", "beamtime", "chemical_formula"]


def get_db_path(dest_root: Path, year: int, month: int) -> Path:
    month_abbr = datetime(year, month, 1).strftime("%b").lower()
    return dest_root / f"{month_abbr}_{year}.db"


def create_database(path: Path) -> Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def is_nexafs_scan_file(file: Path, max_header_lines: int = 50) -> bool:
    try:
        with open(file) as f:
            head = "".join(next(f, "") for _ in range(max_header_lines))
    except OSError:
        return False
    if not re.search(r"Date:\s*\d{1,2}/\d{1,2}/\d{4}", head):
        return False
    markers = ("Scan Type", "Scan Motor", "From File", "Beamline Energy")
    return any(m in head for m in markers)


def catalog_izero_files(directory: Path) -> pd.DataFrame:
    rows = []
    for path in directory.rglob("*.txt"):
        if not is_nexafs_scan_file(path):
            continue
        stem = path.stem.lower()
        if stem.startswith("izero_"):
            parts = path.stem.split("_", 1)
        elif stem.startswith("i0_"):
            parts = path.stem.split("_", 1)
        else:
            continue
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        scan_number = int(parts[1])
        beamtime = pd.Timestamp(
            datetime.fromtimestamp(path.stat().st_birthtime)
        ).normalize().replace(day=1)
        rows.append({"path": path, "scan_number": scan_number, "beamtime": beamtime})
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["path", "scan_number", "beamtime"])
    )


def catalog_calibrant_files(directory: Path) -> pd.DataFrame:
    rows = []
    for path in directory.rglob("*.txt"):
        if not is_nexafs_scan_file(path):
            continue
        parts = path.stem.split("_")
        if len(parts) != 2 or parts[0] != "HOPG" or not parts[1].isdigit():
            continue
        scan_number = int(parts[1])
        beamtime = pd.Timestamp(
            datetime.fromtimestamp(path.stat().st_birthtime)
        ).normalize().replace(day=1)
        rows.append(
            {"path": path, "name": "HOPG", "scan_number": scan_number, "beamtime": beamtime}
        )
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["path", "name", "scan_number", "beamtime"])
    )


def get_beamtime_from_catalogs(
    sample_catalog: pd.DataFrame,
    izero_catalog: pd.DataFrame,
    calibrant_catalog: pd.DataFrame,
) -> tuple[int, int]:
    dfs = [sample_catalog, izero_catalog, calibrant_catalog]
    beams = [
        df["beamtime"]
        for df in dfs
        if not df.empty and "beamtime" in df.columns
    ]
    if not beams:
        return datetime.now().year, datetime.now().month
    min_ts = pd.concat(beams).min()
    return int(min_ts.year), int(min_ts.month)


def _normalize_name(s: str) -> str:
    return (
        re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or str(s).lower()
    )


def apply_tag_map(tag: str, mapping: dict[str, str] | None) -> str:
    normalized = _normalize_name(str(tag)) if str(tag).strip() else ""
    if not mapping:
        return normalized
    return mapping.get(normalized, normalized)


def _keys_from_parse_key(pk: str) -> list[str]:
    return re.findall(r"<([^>]+)>", pk)


def _parse_angle(raw: str | None) -> float | None:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    s = re.sub(r"\s*deg\s*$", "", str(raw).strip(), flags=re.I).strip()
    return float(s) if s else None


def parse_sample(
    parts: list[str],
    keys: list[str],
    birth_time: float,
    formula_map: dict[str, str],
) -> dict[str, Any]:
    if len(parts) == len(keys):
        parsed = dict(zip(keys, parts))
    elif len(parts) == len(keys) - 1 and "version" in keys:
        i = keys.index("version")
        parsed = {
            **dict(zip(keys[:i], parts[:i])),
            "version": 1,
            **dict(zip(keys[i + 1 :], parts[i:])),
        }
    else:
        parsed = dict(zip(keys, parts[: len(keys)]))
    ts = datetime.fromtimestamp(birth_time)
    return {
        "name": parsed["name"],
        "tag": parsed.get("tag", ""),
        "version": (
            int(round(float(parsed["version"])))
            if "version" in parsed
            else 1
        ),
        "beamtime": pd.Timestamp(year=ts.year, month=ts.month, day=1),
        "chemical_formula": formula_map.get(parsed["name"].lower(), ""),
        "angle": _parse_angle(parsed.get("angle")),
        "scan_number": parsed.get("scan_number"),
    }


def build_files_catalog(
    directory: Path,
    formula_map: dict[str, str],
    pk: str,
    tag_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    keys = _keys_from_parse_key(pk)
    rows = []
    for path in directory.rglob("*.txt"):
        if not is_nexafs_scan_file(path):
            continue
        parts = path.stem.split("_")
        if len(parts) < len(keys) and not (
            len(parts) == len(keys) - 1 and "version" in keys
        ):
            continue
        rec = parse_sample(parts, keys, path.stat().st_birthtime, formula_map)
        rec["path"] = path
        rows.append(rec)
    if not rows:
        return pd.DataFrame(
            columns=[
                "path",
                "name",
                "tag",
                "version",
                "beamtime",
                "chemical_formula",
                "angle",
                "scan_number",
            ]
        )
    df = pd.DataFrame(rows)
    df["name"] = df["name"].astype(str).apply(_normalize_name).str.strip()
    df["tag"] = (
        df["tag"]
        .astype(str)
        .apply(lambda t: apply_tag_map(t, tag_map))
        .str.strip()
    )
    return df


def ingest_dataset(file: Path) -> pd.DataFrame:
    with open(file) as f:
        lines = f.readlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if "Time of Day" in line),
        None,
    )
    if header_idx is None:
        raise ValueError(f"Could not find 'Time of Day' header in {file}")
    table_text = "".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(table_text), sep=r"\t", engine="python")
    ts_col = (
        "Time Stamp"
        if "Time Stamp" in df.columns
        else "Time of Day"
        if "Time of Day" in df.columns
        else None
    )
    if ts_col is not None:
        df["Time Stamp"] = pd.to_datetime(df[ts_col], format="mixed")
    else:
        df["Time Stamp"] = pd.to_datetime(file.stat().st_birthtime, unit="s")
    rename = {}
    if "Tey Signal" in df.columns and "TEY signal" not in df.columns:
        rename["Tey Signal"] = "TEY signal"
    if rename:
        df = df.rename(columns=rename)
    df["Beamline Energy"] = df["Beamline Energy"].round(1).astype(float)
    return df[
        [
            "Time Stamp",
            "Beamline Energy",
            "TEY signal",
            "AI 3 Izero",
            "Photodiode",
        ]
    ].copy()


def _izero_before_after(
    izero_df: pd.DataFrame, time_stamp_iso: str
) -> tuple[int | None, int | None]:
    if izero_df.empty:
        return None, None
    df = izero_df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    ts = pd.Timestamp(time_stamp_iso)
    before = df[df["time_stamp"] <= ts]
    after = df[df["time_stamp"] >= ts]
    before_scan_id = (
        int(before.nlargest(1, "time_stamp").iloc[0]["scan_id"])
        if not before.empty
        else None
    )
    after_scan_id = (
        int(after.nsmallest(1, "time_stamp").iloc[0]["scan_id"])
        if not after.empty
        else None
    )
    return before_scan_id, after_scan_id


def build_unique_samples(
    files_catalog: pd.DataFrame,
    calibrant_catalog: pd.DataFrame,
    formula_map: dict[str, str],
) -> pd.DataFrame:
    if not files_catalog.empty:
        out = (
            files_catalog[SAMPLE_COLS]
            .drop_duplicates()
            .reset_index(drop=True)
            .copy()
        )
    else:
        out = pd.DataFrame(columns=SAMPLE_COLS)
    if calibrant_catalog.empty:
        return out
    cal = calibrant_catalog[["name", "beamtime"]].copy()
    cal["name"] = cal["name"].apply(_normalize_name)
    cal = cal.drop_duplicates(subset=["name", "beamtime"]).reset_index(drop=True)
    cal["tag"] = ""
    cal["version"] = 0
    cal["chemical_formula"] = cal["name"].map(formula_map).fillna("")
    cal["beamtime"] = pd.to_datetime(cal["beamtime"]).dt.normalize()
    out = (
        pd.concat([out, cal[SAMPLE_COLS]], ignore_index=True)
        .drop_duplicates(subset=SAMPLE_COLS)
        .reset_index(drop=True)
    )
    return out


def insert_samples(conn: Connection, unique_samples: pd.DataFrame) -> None:
    if unique_samples.empty:
        return
    unique_samples = unique_samples.copy()
    unique_samples["beamtime"] = pd.to_datetime(
        unique_samples["beamtime"]
    ).dt.normalize()
    sample_lookup = pd.read_sql_query(
        "SELECT id, name, tag, version, beamtime FROM sample", conn
    )
    sample_lookup["name"] = sample_lookup["name"].astype(str).str.strip()
    sample_lookup["tag"] = sample_lookup["tag"].astype(str).str.strip()
    sample_lookup["beamtime"] = pd.to_datetime(
        sample_lookup["beamtime"]
    ).dt.normalize()
    merged = unique_samples.merge(
        sample_lookup,
        on=["name", "tag", "version", "beamtime"],
        how="left",
    )
    to_insert = merged.loc[merged["id"].isna(), SAMPLE_COLS].drop_duplicates()
    for _, row in to_insert.iterrows():
        conn.execute(
            "INSERT INTO sample (name, tag, version, beamtime, chemical_formula) VALUES (?, ?, ?, ?, ?)",
            (
                str(row["name"]).strip(),
                str(row["tag"]).strip(),
                int(row["version"]),
                pd.Timestamp(row["beamtime"]).strftime("%Y-%m-%d"),
                row["chemical_formula"],
            ),
        )
    if not to_insert.empty:
        conn.commit()


def load_sample_lookup(conn: Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT id, name, tag, version, beamtime FROM sample", conn
    )
    df["name"] = df["name"].astype(str).str.strip()
    df["tag"] = df["tag"].astype(str).str.strip()
    df["beamtime"] = pd.to_datetime(df["beamtime"]).dt.normalize()
    df["version"] = df["version"].astype(int)
    return df


def ingest_izero_into_db(
    conn: Connection, izero_catalog: pd.DataFrame
) -> None:
    for _, row in izero_catalog.iterrows():
        path = Path(row["path"]) if not isinstance(row["path"], Path) else row["path"]
        scan_id = int(row["scan_number"])
        scan_df = ingest_dataset(path)
        if scan_df.empty:
            continue
        for _, s in scan_df.iterrows():
            time_stamp = pd.Timestamp(s["Time Stamp"]).isoformat()
            conn.execute(
                "INSERT INTO izero (scan_id, beamline_energy, time_stamp, photodiode, ai_3_izero) VALUES (?, ?, ?, ?, ?)",
                (
                    scan_id,
                    float(s["Beamline Energy"]),
                    time_stamp,
                    float(s["Photodiode"]),
                    float(s["AI 3 Izero"]),
                ),
            )
    conn.commit()


def build_rows_to_ingest(
    files_catalog: pd.DataFrame,
    calibrant_catalog: pd.DataFrame,
    sample_lookup: pd.DataFrame,
) -> list[tuple[Path, int, float | None, int]]:
    rows: list[tuple[Path, int, float | None, int]] = []
    if not files_catalog.empty:
        fc = files_catalog.copy()
        fc["version"] = fc["version"].astype(int)
        fc["beamtime"] = pd.to_datetime(fc["beamtime"]).dt.normalize()
        fc["_bk"] = fc["beamtime"].dt.strftime("%Y-%m-%d")
        sl = sample_lookup.copy()
        sl["beamtime"] = pd.to_datetime(sl["beamtime"]).dt.normalize()
        sl["_bk"] = sl["beamtime"].dt.strftime("%Y-%m-%d")
        fc = fc.merge(sl, on=["name", "tag", "version", "_bk"], how="left")
        fc = fc.dropna(subset=["id"])
        for _, row in fc.iterrows():
            if pd.isna(row.get("scan_number")):
                continue
            path = Path(row["path"]) if not isinstance(row["path"], Path) else row["path"]
            rows.append(
                (
                    path,
                    int(row["id"]),
                    float(row["angle"]) if pd.notna(row.get("angle")) else None,
                    int(row["scan_number"]),
                )
            )
    if not calibrant_catalog.empty:
        cal = calibrant_catalog.copy()
        cal["name"] = cal["name"].apply(_normalize_name)
        cal["tag"] = ""
        cal["version"] = 0
        cal["beamtime"] = pd.to_datetime(cal["beamtime"]).dt.normalize()
        cal["_bk"] = cal["beamtime"].dt.strftime("%Y-%m-%d")
        sl = sample_lookup.copy()
        sl["beamtime"] = pd.to_datetime(sl["beamtime"]).dt.normalize()
        sl["_bk"] = sl["beamtime"].dt.strftime("%Y-%m-%d")
        cal = cal.merge(sl, on=["name", "tag", "version", "_bk"], how="left")
        cal = cal.dropna(subset=["id"])
        for _, row in cal.iterrows():
            if pd.isna(row.get("scan_number")):
                continue
            path = Path(row["path"]) if not isinstance(row["path"], Path) else row["path"]
            rows.append((path, int(row["id"]), None, int(row["scan_number"])))
    return rows


def ingest_nexafs_into_db(
    conn: Connection,
    rows_to_ingest: list[tuple[Path, int, float | None, int]],
    izero_df: pd.DataFrame,
) -> None:
    tey_col = "TEY signal"
    for path, sample_id, sample_theta, scan_id in rows_to_ingest:
        scan_df = ingest_dataset(path)
        col = tey_col if tey_col in scan_df.columns else "Tey Signal"
        for _, s in scan_df.iterrows():
            ts_str = pd.Timestamp(s["Time Stamp"]).isoformat()
            before_scan_id, after_scan_id = _izero_before_after(izero_df, ts_str)
            conn.execute(
                """INSERT INTO nexafs (
                    scan_id, sample_id, time_stamp, sample_theta, beamline_energy,
                    tey_signal, ai_3_izero, photodiode, izero_before_scan_id, izero_after_scan_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scan_id,
                    sample_id,
                    ts_str,
                    sample_theta,
                    float(s["Beamline Energy"]),
                    float(s[col]),
                    float(s["AI 3 Izero"]),
                    float(s["Photodiode"]),
                    before_scan_id,
                    after_scan_id,
                ),
            )
    conn.commit()


def run_pipeline(
    data_root: Path,
    dest_root: Path,
    formula_map: dict[str, str],
    pk: str,
    tag_map: dict[str, str] | None = None,
) -> tuple[Connection | None, pd.DataFrame, pd.DataFrame]:
    izero_catalog = catalog_izero_files(data_root)
    calibrant_catalog = catalog_calibrant_files(data_root)
    files_catalog = build_files_catalog(data_root, formula_map, pk, tag_map)
    unique_samples = build_unique_samples(
        files_catalog, calibrant_catalog, formula_map
    )
    if unique_samples.empty:
        return None, files_catalog, unique_samples
    unique_samples["beamtime"] = pd.to_datetime(
        unique_samples["beamtime"]
    ).dt.normalize()
    year, month = get_beamtime_from_catalogs(
        files_catalog, izero_catalog, calibrant_catalog
    )
    db_path = get_db_path(dest_root, year, month)
    conn = create_database(db_path)
    insert_samples(conn, unique_samples)
    sample_lookup = load_sample_lookup(conn)
    ingest_izero_into_db(conn, izero_catalog)
    izero_df = pd.read_sql_query("SELECT id, time_stamp, scan_id FROM izero", conn)
    rows_to_ingest = build_rows_to_ingest(
        files_catalog, calibrant_catalog, sample_lookup
    )
    ingest_nexafs_into_db(conn, rows_to_ingest, izero_df)
    return conn, files_catalog, unique_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NEXAFS beamtime pipeline: discover files, create DB, ingest sample/izero/nexafs."
    )
    parser.add_argument(
        "data_root",
        type=Path,
        nargs="?",
        default=Path(
            "/Users/hduva/Library/CloudStorage/OneDrive-SharedLibraries-WashingtonStateUniversity(email.wsu.edu)/Carbon Lab Research Group - Documents/Synchrotron Logistics and Data/ALS - Berkeley/Data/BL1101/2025Oct/TAD KOGA/renamed"
        ),
        help="Beamtime directory containing NEXAFS .txt files",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("/Users/hduva/projects/als-nexafs"),
        help="Destination directory for mon_yyyy.db",
    )
    args = parser.parse_args()
    formula_map: dict[str, str] = {"ps": "C8H8", "hopg": "C"}
    parse_key = "<name>_<tag>_<version>_<angle>_<scan_number>"
    tag_map: dict[str, str] = {}
    conn, files_catalog, unique_samples = run_pipeline(
        args.data_root, args.dest, formula_map, parse_key, tag_map
    )
    if conn is None:
        print("No sample/calibrant files found; no database created.")
        return
    sample_count = pd.read_sql_query(
        "SELECT COUNT(*) as n FROM sample", conn
    ).iloc[0, 0]
    izero_count = pd.read_sql_query(
        "SELECT COUNT(*) as n FROM izero", conn
    ).iloc[0, 0]
    nexafs_count = pd.read_sql_query(
        "SELECT COUNT(*) as n FROM nexafs", conn
    ).iloc[0, 0]
    print(f"DB: sample={sample_count}, izero={izero_count}, nexafs={nexafs_count}")
    samples = pd.read_sql_query(
        "SELECT id, name, tag, version FROM sample ORDER BY id", conn
    )
    print("Samples:", samples.to_string(index=False))
    nexafs_by_sample = pd.read_sql_query(
        "SELECT sample_id, COUNT(*) as nexafs_rows FROM nexafs GROUP BY sample_id",
        conn,
    )
    print("NEXAFS rows per sample:", nexafs_by_sample.to_string(index=False))


if __name__ == "__main__":
    main()
