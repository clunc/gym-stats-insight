"""Weekly view page."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app_utils import (
    compound_exercises,
    compute_summary,
    load_app_data,
    progression_event_summary,
    progression_status,
    rep_cap_for_exercise,
    per_day_volume,
    stagnation_flag_in_period,
    weekly_breakdown,
    weekly_volume_context,
    workout_date_bounds,
)

st.title("Weekly View")

data = load_app_data()
if not data:
    st.stop()

_, _, df = data

_, latest = workout_date_bounds(df)
if latest is None:
    st.info("No workouts available for weekly summaries.")
    st.stop()

option = st.radio(
    "Timeframe",
    ("Last 7 days (rolling)", "Current calendar week"),
    horizontal=True,
)
end_date = latest.normalize()
if option == "Current calendar week":
    start_date = end_date.to_period("W").start_time
    label = "This calendar week"
else:
    start_date = end_date - pd.Timedelta(days=6)
    label = "Last 7 days"

st.caption(f"{label} (ending {end_date.date()})")

start_str = start_date.date().isoformat()
end_str = end_date.date().isoformat()
weekly_df = df[
    (df["type"] == "workout")
    & (df["date"] >= start_str)
    & (df["date"] <= end_str)
]
exercises = sorted(df[df["type"] == "workout"]["exercise"].unique())

weekly_summary = compute_summary(weekly_df) if not weekly_df.empty else {
    "total_sessions": 0,
    "total_sets": 0,
    "total_reps": 0,
    "total_volume": 0.0,
}

w1, w2, w3, w4 = st.columns(4)
w1.metric("Sessions this period", weekly_summary["total_sessions"])
w2.metric("Total Sets", weekly_summary["total_sets"])
w3.metric("Total Reps", weekly_summary["total_reps"])
w4.metric("Total Volume", f"{weekly_summary['total_volume']:.0f} kg")

st.subheader("Progression Events (This Period)")
period_total, period_table = progression_event_summary(weekly_df, exercises)
st.metric("Progression Events", period_total)
if period_total == 0:
    st.caption("No progression events in this period.")
else:
    st.dataframe(period_table, use_container_width=True)

st.subheader("Session Volume (Weekly)")
if weekly_df.empty:
    st.info("No sessions recorded in this period.")
else:
    weekly_volume_df = per_day_volume(weekly_df)
    weekly_chart = (
        alt.Chart(weekly_volume_df)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("volume:Q", title="Volume (kg·reps)"),
            tooltip=["date:T", "volume:Q"],
        )
        .properties(height=240)
    )
    st.altair_chart(weekly_chart, use_container_width=True)

st.subheader("Per-Exercise Weekly Breakdown")
breakdown = weekly_breakdown(weekly_df)
if breakdown.empty:
    st.caption("No exercise data for this period.")
else:
    st.dataframe(breakdown, use_container_width=True)

st.subheader("Per-Exercise Fatigue Context (Compounds Only)")
compounds = compound_exercises(exercises)
if not compounds:
    st.caption("No compounds tagged; using name-based matching (deadlift/squat/bench/press/row).")
compound_rows = []
for exercise in compounds:
    ex_df = df[(df["type"] == "workout") & (df["exercise"] == exercise)]
    period_ex_df = weekly_df[weekly_df["exercise"] == exercise]
    cap = rep_cap_for_exercise(exercise)
    context = weekly_volume_context(ex_df)
    latest_date = period_ex_df["date"].max() if not period_ex_df.empty else None
    latest_session = period_ex_df[period_ex_df["date"] == latest_date] if latest_date else None
    ready = False
    if latest_session is not None and not latest_session.empty:
        ready = progression_status(latest_session, cap)["ready"]
    stagnant = stagnation_flag_in_period(ex_df, cap, start_str, end_str)
    compound_rows.append(
        {
            "exercise": exercise,
            "weekly_volume": f"{context['weekly_volume']:.0f}",
            "rolling_avg_3w": f"{context['rolling_avg']:.0f}",
            "trend": context["trend"],
            "progression_ready": "Yes" if latest_date and ready else "No",
            "stagnation_flag": "Yes" if stagnant else "No",
        }
    )
if compound_rows:
    st.dataframe(pd.DataFrame(compound_rows), use_container_width=True)
else:
    st.caption("No compound exercise history available.")
