"""EMS Response-Time Analytics Dashboard (NYC Open Data).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import ems_pipeline as ems

st.set_page_config(page_title="EMS Response Times", page_icon="🚑", layout="wide")

DB = Path(os.getenv("EMS_DB", ems.DEFAULT_DB))
ACCENT = "#C8102E"
BOROUGH_ORDER = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]


# --------------------------------------------------------------------------
# Data: build the database on first start (real data, offline sample as fallback)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_db(db: str) -> str:
    if Path(db).exists():
        return "existing"
    try:
        token = st.secrets.get("NYC_APP_TOKEN") if hasattr(st, "secrets") else None
    except Exception:
        token = None
    try:
        ems.run(limit=int(os.getenv("EMS_LIMIT", 100_000)), db_path=db, app_token=token)
        return "downloaded"
    except Exception:
        ems.run(db_path=db, sample=True)
        return "sample"


with st.spinner("Downloading the latest EMS incidents from NYC Open Data (first start only)…"):
    status = ensure_db(str(DB))

info = ems.meta(DB)
is_sample = info.get("source") == "synthetic_sample"
min_d, max_d = date.fromisoformat(info["min_date"]), date.fromisoformat(info["max_date"])


@st.cache_data(show_spinner=False)
def load(filters: tuple, loaded_at: str) -> pd.DataFrame:
    return ems.query_incidents(dict(filters), DB)


@st.cache_data(show_spinner=False)
def call_types(loaded_at: str) -> list[str]:
    return ems.all_call_types(DB)


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🚑 EMS Response Times")
    st.caption("New York City · EMS Incident Dispatch Data")
    default_start = max(min_d, max_d - timedelta(days=30))
    period = st.date_input("Period", (default_start, max_d), min_value=min_d, max_value=max_d)
    start, end = (period if isinstance(period, tuple) and len(period) == 2 else (min_d, max_d))
    boroughs = st.multiselect("Boroughs", BOROUGH_ORDER, default=BOROUGH_ORDER)
    life = st.toggle("Life-threatening only", value=False, help="NYC severity levels 1–3.")
    types = st.multiselect("Call types", call_types(info["loaded_at"]), placeholder="All call types")
    target = st.slider("Response-time target (min)", 4, 20, 10,
                       help="Share of incidents where the first unit arrived within this time.")
    st.divider()
    st.caption(f"Data: {int(info['rows']):,} incidents, {info['min_date']} to {info['max_date']}  \n"
               f"Loaded {info['loaded_at'][:16].replace('T', ' ')} UTC")
    st.caption(f"[Source: NYC Open Data]({ems.DATASET_PAGE})")

filters = {"start": start.isoformat(), "end": end.isoformat(), "boroughs": tuple(boroughs) or tuple(BOROUGH_ORDER),
           "life_threatening_only": life, "call_types": tuple(types)}
df = load(tuple(sorted(filters.items())), info["loaded_at"])

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.header("How fast does an ambulance arrive in New York City?")
if is_sample:
    st.warning("Showing the **synthetic demo sample** (invented data) because NYC Open Data could not be "
               "reached. Run `python ems_pipeline.py` with internet access to load real incidents.", icon="⚠️")
st.caption("Response time = from call creation to the first EMS unit on scene. Only incidents NYC marks with a valid "
           "response time are included.")

if df.empty:
    st.info("No incidents match these filters. Widen the period or select more boroughs.")
    st.stop()

k = ems.kpis(df, target)
c = st.columns(6)
c[0].metric("Incidents", f"{k['incidents']:,}")
c[1].metric("Median response", f"{k['median']:.1f} min")
c[2].metric("90th percentile", f"{k['p90']:.1f} min", help="9 in 10 incidents were reached within this time.")
c[3].metric(f"Within {target} min", f"{k['within_target']:.0%}")
c[4].metric("Median dispatch", f"{k['median_dispatch']:.1f} min", help="From call creation to unit assignment.")
c[5].metric("Median travel", f"{k['median_travel']:.1f} min", help="Driving time from activation to the scene.")

# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("By borough")
    bp = ems.borough_percentiles(df).melt("borough", var_name="measure", value_name="minutes")
    bp["measure"] = bp["measure"].map({"median": "Median", "p90": "90th percentile"})
    chart = (alt.Chart(bp).mark_bar(cornerRadiusEnd=3)
             .encode(y=alt.Y("borough:N", sort=BOROUGH_ORDER, title=None),
                     x=alt.X("minutes:Q", title="Response time (min)"),
                     yOffset="measure:N",
                     color=alt.Color("measure:N", sort=["Median", "90th percentile"], scale=alt.Scale(range=["#F2A6B0", ACCENT]),
                                     legend=alt.Legend(orient="bottom", title=None)),
                     tooltip=["borough", "measure", alt.Tooltip("minutes:Q", format=".1f")])
             .properties(height=280))
    rule = alt.Chart(pd.DataFrame({"t": [target]})).mark_rule(strokeDash=[4, 4]).encode(x="t:Q")
    st.altair_chart(chart + rule, use_container_width=True)

with right:
    st.subheader("By hour of day")
    bh = ems.by_hour(df)
    base = alt.Chart(bh).encode(x=alt.X("hour:O", title="Hour of day"))
    bars = base.mark_bar(color="#D9DDE3").encode(y=alt.Y("incidents:Q", title="Incidents"),
                                                  tooltip=["hour", "incidents"])
    line = base.mark_line(color=ACCENT, point=alt.OverlayMarkDef(color=ACCENT)).encode(
        y=alt.Y("median_response_min:Q", title="Median response (min)"),
        tooltip=["hour", alt.Tooltip("median_response_min:Q", format=".1f")])
    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=280),
                    use_container_width=True)

st.subheader("When are responses slowest?")
st.caption("Median response time by weekday and hour. Darker = slower.")
hm = ems.weekday_hour_matrix(df)
heat = (alt.Chart(hm).mark_rect()
        .encode(x=alt.X("hour:O", title="Hour of day"),
                y=alt.Y("weekday_name:N", sort=ems.WEEKDAYS, title=None),
                color=alt.Color("response_min:Q", title="Median (min)", scale=alt.Scale(scheme="reds")),
                tooltip=["weekday_name", "hour", alt.Tooltip("response_min:Q", format=".1f")])
        .properties(height=230))
st.altair_chart(heat, use_container_width=True)

l2, r2 = st.columns(2)
with l2:
    st.subheader("Most frequent call types")
    tc = ems.top_call_types(filters, 10, DB)
    st.dataframe(tc.rename(columns={"call_type": "Call type", "incidents": "Incidents",
                                    "avg_response_min": "Avg response (min)"}),
                 hide_index=True, use_container_width=True)
with r2:
    st.subheader("Response-time distribution")
    hist = (alt.Chart(df[df["response_min"] <= 40]).mark_bar(color=ACCENT, opacity=0.85)
            .encode(x=alt.X("response_min:Q", bin=alt.Bin(step=1), title="Response time (min)"),
                    y=alt.Y("count():Q", title="Incidents"))
            .properties(height=300))
    st.altair_chart(hist + rule, use_container_width=True)

st.caption("Built with Python, Pandas, SQLite and Streamlit · Data: NYC Open Data, EMS Incident Dispatch Data (76xm-jjuj)")
