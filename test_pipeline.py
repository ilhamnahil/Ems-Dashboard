import pandas as pd
import pytest

import ems_pipeline as ems


def raw_rows():
    base = {
        "initial_call_type": "SICK", "final_call_type": "DIFFBR",
        "initial_severity_level_code": "6", "final_severity_level_code": "2",
        "valid_dispatch_rspns_time_indc": "Y", "dispatch_response_seconds_qy": "120",
        "valid_incident_rspns_time_indc": "Y", "incident_response_seconds_qy": "540",
        "incident_travel_tm_seconds_qy": "360", "held_indicator": "N",
        "borough": "RICHMOND / STATEN ISLAND", "incident_dispatch_area": "S1", "zipcode": "10301",
    }
    return pd.DataFrame([
        {**base, "incident_id": "1", "incident_datetime": "2026-09-01T08:15:00.000"},
        {**base, "incident_id": "1", "incident_datetime": "2026-09-01T08:15:00.000"},            # duplicate
        {**base, "incident_id": "2", "incident_datetime": "2026-09-02T23:05:00.000",
         "valid_incident_rspns_time_indc": "N"},                                                 # invalid time
        {**base, "incident_id": "3", "incident_datetime": "not a date"},                          # bad timestamp
        {**base, "incident_id": "4", "incident_datetime": "2026-09-03T12:00:00.000",
         "incident_response_seconds_qy": "99999"},                                               # > 3 h outlier
        {**base, "incident_id": "5", "incident_datetime": "2026-09-06T17:30:00.000", "borough": "BRONX",
         "final_severity_level_code": "7", "final_call_type": None, "valid_dispatch_rspns_time_indc": "N"},
    ])


def test_transform_cleans_and_derives_fields():
    out = ems.transform(raw_rows())
    assert list(out["incident_id"]) == ["1", "5"]
    first, second = out.iloc[0], out.iloc[1]
    assert first["borough"] == "Staten Island" and first["hour"] == 8 and first["weekday"] == 1  # Tuesday
    assert first["response_min"] == 9.0 and first["travel_min"] == 6.0 and first["dispatch_min"] == 2.0
    assert first["call_type"] == "DIFFBR" and first["life_threatening"] == 1
    assert second["call_type"] == "SICK"            # falls back to the initial call type
    assert second["life_threatening"] == 0
    assert pd.isna(second["dispatch_min"])          # invalid dispatch time is dropped, not zeroed


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "ems.db"
    ems.load(ems.transform(raw_rows()), path, source="test")
    return path


def test_load_writes_rows_and_meta(db):
    m = ems.meta(db)
    assert m["source"] == "test" and m["rows"] == "2"
    assert (m["min_date"], m["max_date"]) == ("2026-09-01", "2026-09-06")


def test_queries_apply_filters(db):
    f = {"start": "2026-09-01", "end": "2026-09-30"}
    assert len(ems.query_incidents(f, db)) == 2
    assert len(ems.query_incidents({**f, "boroughs": ("Bronx",)}, db)) == 1
    assert len(ems.query_incidents({**f, "life_threatening_only": True}, db)) == 1
    assert len(ems.query_incidents({**f, "call_types": ("SICK",)}, db)) == 1
    assert len(ems.query_incidents({"start": "2026-10-01", "end": "2026-10-31"}, db)) == 0
    bc = ems.borough_counts(f, db)
    assert set(bc["borough"]) == {"Staten Island", "Bronx"}
    assert ems.top_call_types(f, 1, db)["incidents"].iloc[0] == 1


def test_kpis():
    df = pd.DataFrame({"response_min": [4, 6, 8, 10, 30], "dispatch_min": [1, 1, 2, 2, None],
                       "travel_min": [3, 4, 5, 6, 7]})
    k = ems.kpis(df, target_min=8)
    assert k["incidents"] == 5 and k["median"] == 8
    assert k["within_target"] == pytest.approx(0.6)
    assert k["median_dispatch"] == 1.5
    assert ems.kpis(df.iloc[0:0])["median"] is None


def test_by_hour_has_all_24_hours():
    df = pd.DataFrame({"hour": [0, 0, 13], "response_min": [5, 7, 9]})
    bh = ems.by_hour(df)
    assert len(bh) == 24 and bh.loc[0, "median_response_min"] == 6 and pd.isna(bh.loc[5, "incidents"])


def test_extract_pages_through_api():
    pages = [[{"incident_id": str(i)} for i in range(3)], [{"incident_id": "9"}]]

    class Resp:
        def __init__(self, data): self.data = data
        def raise_for_status(self): pass
        def json(self): return self.data

    class Session:
        def __init__(self): self.calls = []
        def get(self, url, params, headers, timeout):
            self.calls.append(params)
            return Resp(pages[len(self.calls) - 1])

    s = Session()
    old = ems.PAGE_SIZE
    ems.PAGE_SIZE = 3
    try:
        df = ems.extract(limit=10, session=s)
    finally:
        ems.PAGE_SIZE = old
    assert len(df) == 4 and len(s.calls) == 2
    assert s.calls[1]["$offset"] == 3 and s.calls[0]["$order"] == "incident_datetime DESC"
