"""Generate a SYNTHETIC demo sample in the raw NYC API format.

The numbers are invented (seeded random data shaped like the real dataset) so the
dashboard and tests run offline. Real analysis uses `python ems_pipeline.py`,
which downloads the actual NYC Open Data records.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("data") / "sample_synthetic.csv"

BOROUGHS = {  # name: (share of calls, travel-time factor)
    "BROOKLYN": (0.31, 1.00), "MANHATTAN": (0.24, 0.90), "BRONX": (0.21, 1.05),
    "QUEENS": (0.20, 1.15), "RICHMOND / STATEN ISLAND": (0.04, 1.20),
}
CALL_TYPES = {  # code: (share, severity)
    "SICK": (0.20, 6), "INJURY": (0.15, 5), "DIFFBR": (0.10, 2), "EDP": (0.09, 4),
    "DRUG": (0.07, 3), "UNC": (0.06, 2), "CARD": (0.05, 3), "ABDPN": (0.05, 5),
    "ALTMEN": (0.05, 3), "MVAINJ": (0.04, 4), "ARREST": (0.02, 1), "CVA": (0.02, 2),
    "OTHER": (0.10, 7),
}
HOURLY = np.array([3.0, 2.5, 2.2, 2.0, 1.9, 2.0, 2.6, 3.4, 4.2, 4.8, 5.1, 5.3,
                   5.4, 5.4, 5.3, 5.3, 5.2, 5.2, 5.1, 4.9, 4.6, 4.2, 3.8, 3.4])


def make(n: int = 40_000, days: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = datetime(2026, 9, 30, 23, 59)
    start = end - timedelta(days=days)
    day_idx = rng.integers(0, days, n)
    hour = rng.choice(24, n, p=HOURLY / HOURLY.sum())
    ts = [start + timedelta(days=int(d), hours=int(h), seconds=int(s))
          for d, h, s in zip(day_idx, hour, rng.integers(0, 3600, n))]

    b_names = list(BOROUGHS)
    borough = rng.choice(b_names, n, p=[BOROUGHS[b][0] for b in b_names])
    c_names = list(CALL_TYPES)
    shares = np.array([CALL_TYPES[c][0] for c in c_names])
    call = rng.choice(c_names, n, p=shares / shares.sum())
    sev = np.array([CALL_TYPES[c][1] for c in call])

    busy = np.interp(hour, range(24), HOURLY) / HOURLY.max()           # busier hours -> slower dispatch
    lt = sev <= 3
    dispatch = rng.lognormal(np.log(np.where(lt, 45, 150)) + 0.6 * busy, 0.7)   # seconds
    activation = rng.lognormal(np.log(60), 0.4, n)
    travel_factor = np.array([BOROUGHS[b][1] for b in borough])
    travel = rng.lognormal(np.log(300 * travel_factor), 0.45)
    response = dispatch + activation + travel

    df = pd.DataFrame({
        "incident_id": np.arange(900_000_000, 900_000_000 + n).astype(str),
        "incident_datetime": [t.strftime("%Y-%m-%dT%H:%M:%S.000") for t in ts],
        "initial_call_type": call,
        "final_call_type": call,
        "initial_severity_level_code": sev.astype(str),
        "final_severity_level_code": sev.astype(str),
        "valid_dispatch_rspns_time_indc": "Y",
        "dispatch_response_seconds_qy": dispatch.round().astype(int).astype(str),
        "valid_incident_rspns_time_indc": "Y",
        "incident_response_seconds_qy": response.round().astype(int).astype(str),
        "incident_travel_tm_seconds_qy": travel.round().astype(int).astype(str),
        "held_indicator": np.where(dispatch > 600, "Y", "N"),
        "borough": borough,
        "incident_dispatch_area": [b[0] + str(rng.integers(1, 9)) for b in borough],
        "zipcode": "",
    })
    # Like the real data, some incidents have no valid response time (cancelled, no unit on scene).
    invalid = rng.random(n) < 0.06
    df.loc[invalid, "valid_incident_rspns_time_indc"] = "N"
    df.loc[invalid, "incident_response_seconds_qy"] = ""
    return df.sort_values("incident_datetime", ascending=False)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    make().to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
