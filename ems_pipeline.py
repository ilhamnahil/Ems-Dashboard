"""ETL pipeline for NYC EMS Incident Dispatch Data.

Extract  : NYC Open Data Socrata API (dataset 76xm-jjuj), newest incidents first
Transform: clean with Pandas, keep valid response times, derive analysis fields
Load     : SQLite database (`incidents` table + `meta` table)

Usage:
    python ems_pipeline.py                 # download latest 100,000 incidents
    python ems_pipeline.py --limit 300000  # more history
    python ems_pipeline.py --sample        # offline: build from the synthetic demo sample
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

API_URL = "https://data.cityofnewyork.us/resource/76xm-jjuj.json"
DATASET_PAGE = "https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj"
DEFAULT_DB = Path(__file__).with_name("ems.db")
SAMPLE_CSV = Path(__file__).with_name("data") / "sample_synthetic.csv"
PAGE_SIZE = 50_000

RAW_COLUMNS = [
    "incident_id", "incident_datetime", "initial_call_type", "final_call_type",
    "initial_severity_level_code", "final_severity_level_code",
    "valid_dispatch_rspns_time_indc", "dispatch_response_seconds_qy",
    "valid_incident_rspns_time_indc", "incident_response_seconds_qy",
    "incident_travel_tm_seconds_qy", "held_indicator", "borough",
    "incident_dispatch_area", "zipcode",
]

BOROUGHS = {
    "BRONX": "Bronx", "BROOKLYN": "Brooklyn", "MANHATTAN": "Manhattan",
    "QUEENS": "Queens", "RICHMOND / STATEN ISLAND": "Staten Island",
    "STATEN ISLAND": "Staten Island",
}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
def extract(limit: int = 100_000, app_token: str | None = None, session=None) -> pd.DataFrame:
    """Download the newest `limit` incidents from the NYC Open Data API, page by page."""
    import requests

    s = session or requests.Session()
    headers = {"X-App-Token": app_token} if app_token else {}
    frames, offset = [], 0
    while offset < limit:
        size = min(PAGE_SIZE, limit - offset)
        params = {
            "$select": ",".join(RAW_COLUMNS),
            "$order": "incident_datetime DESC",
            "$limit": size,
            "$offset": offset,
        }
        for attempt in range(3):
            try:
                r = s.get(API_URL, params=params, headers=headers, timeout=60)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        page = pd.DataFrame(r.json())
        if page.empty:
            break
        frames.append(page)
        offset += len(page)
        if len(page) < size:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_COLUMNS)


def extract_sample(path: Path = SAMPLE_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw API rows into one analysis-ready row per incident."""
    df = raw.copy()
    for col in RAW_COLUMNS:
        if col not in df:
            df[col] = np.nan

    df["incident_datetime"] = pd.to_datetime(df["incident_datetime"], errors="coerce")
    for col in ["incident_response_seconds_qy", "dispatch_response_seconds_qy", "incident_travel_tm_seconds_qy",
                "final_severity_level_code"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only incidents NYC flags as having a valid response time, and drop impossible values.
    valid = (
        df["incident_datetime"].notna()
        & (df["valid_incident_rspns_time_indc"] == "Y")
        & df["incident_response_seconds_qy"].between(1, 3 * 3600)
    )
    df = df[valid].drop_duplicates("incident_id")

    out = pd.DataFrame({
        "incident_id": df["incident_id"].astype(str),
        "incident_datetime": df["incident_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "date": df["incident_datetime"].dt.strftime("%Y-%m-%d"),
        "hour": df["incident_datetime"].dt.hour,
        "weekday": df["incident_datetime"].dt.weekday,  # 0 = Monday
        "borough": df["borough"].map(BOROUGHS).fillna("Unknown"),
        "dispatch_area": df["incident_dispatch_area"].fillna(""),
        "call_type": df["final_call_type"].fillna(df["initial_call_type"]).fillna("UNKNOWN").str.strip(),
        "severity": df["final_severity_level_code"].fillna(0).astype(int),
        "response_min": (df["incident_response_seconds_qy"] / 60).round(2),
        "dispatch_min": (df["dispatch_response_seconds_qy"].where(df["valid_dispatch_rspns_time_indc"] == "Y") / 60).round(2),
        "travel_min": (df["incident_travel_tm_seconds_qy"] / 60).round(2),
        "held": (df["held_indicator"] == "Y").astype(int),
    })
    # NYC EMS severity 1-3 = life-threatening (segments 1-3); 4-8 = non-life-threatening.
    out["life_threatening"] = out["severity"].between(1, 3).astype(int)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE incidents (
    incident_id      TEXT PRIMARY KEY,
    incident_datetime TEXT NOT NULL,
    date             TEXT NOT NULL,
    hour             INTEGER NOT NULL,
    weekday          INTEGER NOT NULL,
    borough          TEXT NOT NULL,
    dispatch_area    TEXT,
    call_type        TEXT NOT NULL,
    severity         INTEGER,
    life_threatening INTEGER NOT NULL,
    response_min     REAL NOT NULL,
    dispatch_min     REAL,
    travel_min       REAL,
    held             INTEGER NOT NULL
);
CREATE INDEX idx_date ON incidents(date);
CREATE INDEX idx_borough ON incidents(borough);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def load(df: pd.DataFrame, db_path: Path | str = DEFAULT_DB, source: str = "nyc_open_data") -> int:
    """Replace the database contents with `df`. Returns the number of rows loaded."""
    db_path = Path(db_path)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(tmp) as con:
        con.executescript(SCHEMA)
        df.to_sql("incidents", con, if_exists="append", index=False)
        meta = {
            "source": source,
            "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": str(len(df)),
            "min_date": df["date"].min() if len(df) else "",
            "max_date": df["date"].max() if len(df) else "",
        }
        con.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
    tmp.replace(db_path)  # atomic swap: the app never sees a half-written database
    return len(df)


def run(limit: int = 100_000, db_path: Path | str = DEFAULT_DB, sample: bool = False,
        app_token: str | None = None) -> int:
    if sample:
        return load(transform(extract_sample()), db_path, source="synthetic_sample")
    return load(transform(extract(limit, app_token)), db_path, source="nyc_open_data")


# ---------------------------------------------------------------------------
# Queries used by the dashboard (SQL does the filtering and grouping)
# ---------------------------------------------------------------------------
def meta(db_path: Path | str = DEFAULT_DB) -> dict:
    with sqlite3.connect(db_path) as con:
        return dict(con.execute("SELECT key, value FROM meta").fetchall())


def _where(filters: dict) -> tuple[str, list]:
    clauses, params = ["date BETWEEN ? AND ?"], [filters["start"], filters["end"]]
    if filters.get("boroughs"):
        clauses.append(f"borough IN ({','.join('?' * len(filters['boroughs']))})")
        params += list(filters["boroughs"])
    if filters.get("life_threatening_only"):
        clauses.append("life_threatening = 1")
    if filters.get("call_types"):
        clauses.append(f"call_type IN ({','.join('?' * len(filters['call_types']))})")
        params += list(filters["call_types"])
    return " AND ".join(clauses), params


def query_incidents(filters: dict, db_path: Path | str = DEFAULT_DB) -> pd.DataFrame:
    """Rows matching the filters (only the columns the dashboard needs)."""
    where, params = _where(filters)
    sql = f"""SELECT date, hour, weekday, borough, call_type, severity, life_threatening,
                     response_min, dispatch_min, travel_min
              FROM incidents WHERE {where}"""
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con, params=params)


def borough_counts(filters: dict, db_path: Path | str = DEFAULT_DB) -> pd.DataFrame:
    where, params = _where(filters)
    sql = f"""SELECT borough,
                     COUNT(*)                         AS incidents,
                     ROUND(AVG(response_min), 1)      AS avg_response_min,
                     ROUND(100.0 * SUM(life_threatening) / COUNT(*), 1) AS pct_life_threatening
              FROM incidents WHERE {where}
              GROUP BY borough ORDER BY incidents DESC"""
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con, params=params)


def top_call_types(filters: dict, n: int = 10, db_path: Path | str = DEFAULT_DB) -> pd.DataFrame:
    where, params = _where(filters)
    sql = f"""SELECT call_type, COUNT(*) AS incidents, ROUND(AVG(response_min), 1) AS avg_response_min
              FROM incidents WHERE {where}
              GROUP BY call_type ORDER BY incidents DESC LIMIT ?"""
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con, params=params + [n])


def all_call_types(db_path: Path | str = DEFAULT_DB) -> list[str]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT call_type FROM incidents GROUP BY call_type ORDER BY COUNT(*) DESC").fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Metrics (Pandas: percentiles are not available in SQLite)
# ---------------------------------------------------------------------------
def kpis(df: pd.DataFrame, target_min: float = 10.0) -> dict:
    if df.empty:
        return {"incidents": 0, "median": None, "p90": None, "within_target": None,
                "median_dispatch": None, "median_travel": None}
    r = df["response_min"]
    return {
        "incidents": int(len(df)),
        "median": float(r.median()),
        "p90": float(r.quantile(0.9)),
        "within_target": float((r <= target_min).mean()),
        "median_dispatch": float(df["dispatch_min"].median()) if df["dispatch_min"].notna().any() else None,
        "median_travel": float(df["travel_min"].median()) if df["travel_min"].notna().any() else None,
    }


def by_hour(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("hour")["response_min"]
    out = pd.DataFrame({"incidents": g.size(), "median_response_min": g.median().round(2),
                        "p90_response_min": g.quantile(0.9).round(2)})
    return out.reindex(range(24)).reset_index().rename(columns={"index": "hour"})


def weekday_hour_matrix(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby(["weekday", "hour"])["response_min"].median().round(2).reset_index()
    m["weekday_name"] = m["weekday"].map(dict(enumerate(WEEKDAYS)))
    return m


def borough_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("borough")["response_min"]
    return pd.DataFrame({"median": g.median(), "p90": g.quantile(0.9)}).round(2).reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the EMS SQLite database.")
    ap.add_argument("--limit", type=int, default=100_000, help="newest incidents to download")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--sample", action="store_true", help="use the bundled synthetic sample (offline)")
    ap.add_argument("--app-token", default=None, help="optional NYC Open Data app token (higher rate limits)")
    a = ap.parse_args()
    n = run(a.limit, a.db, a.sample, a.app_token)
    print(f"Loaded {n:,} incidents into {a.db}")
