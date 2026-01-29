"""Year-to-date view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app_utils import load_app_data, per_day_volume, workout_date_bounds, ytd_volume

st.title("YTD View")

data = load_app_data()
if not data:
    st.stop()

_, _, df = data

_, last_date = workout_date_bounds(df)
if last_date is None:
    st.info("No workouts available for YTD summaries.")
    st.stop()

year = last_date.year
start_date = pd.Timestamp(year=year, month=1, day=1, tz=last_date.tz)
end_date = last_date.normalize()
start_str = start_date.date().isoformat()
end_str = end_date.date().isoformat()

ytd_df = df[
    (df["type"] == "workout")
    & (df["date"] >= start_str)
    & (df["date"] <= end_str)
]

st.metric(f"YTD Volume ({year})", f"{ytd_volume(df, year):.0f} kg")

st.subheader("YTD Session Volume")
if ytd_df.empty:
    st.caption("No sessions recorded year-to-date.")
else:
    ytd_volume_df = per_day_volume(ytd_df)
    chart = (
        alt.Chart(ytd_volume_df)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("volume:Q", title="Volume (kg·reps)"),
            tooltip=["date:T", "volume:Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)
