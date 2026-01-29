"""Overview (All Time) page."""

from __future__ import annotations

import streamlit as st

from app_utils import (
    absolute_progression_kg,
    compute_summary,
    load_app_data,
    latest_1rm_table,
    progression_event_summary,
    workout_date_bounds,
)

st.title("Overview (All Time)")

data = load_app_data()
if not data:
    st.stop()

_, template, df = data

first_date, last_date = workout_date_bounds(df)
if first_date is not None:
    st.caption(f"Since first logged workout on {first_date.date()}")

summary = compute_summary(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sessions", summary["total_sessions"])
col2.metric("Total Sets", summary["total_sets"])
col3.metric("Total Reps", summary["total_reps"])
col4.metric("Total Volume", f"{summary['total_volume']:.0f} kg")

st.subheader("Current 1RM per Exercise")
latest_1rm = latest_1rm_table(df)
if latest_1rm.empty:
    st.caption("No 1RM estimates available.")
else:
    st.dataframe(latest_1rm, use_container_width=True)

st.subheader("Progression Events (All Time)")
exercises = sorted(df[df["type"] == "workout"]["exercise"].unique())
total_events, event_table = progression_event_summary(df, exercises)
if total_events == 0:
    st.caption("No progression events recorded.")
else:
    if first_date is not None:
        st.caption(f"Progression events since {first_date.date()}")
    progress_df = absolute_progression_kg(df)
    table = event_table.merge(progress_df, on="exercise", how="left")
    st.dataframe(table, use_container_width=True)

st.subheader("Sick Days")
sick = df[df["type"] == "sick"]
if sick.empty:
    st.caption("No sick days logged.")
else:
    sick_dates = sick["date"].drop_duplicates().sort_values().tolist()
    st.write(", ".join(sick_dates))
