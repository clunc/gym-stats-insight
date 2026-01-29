"""Monthly volume view."""

from __future__ import annotations

import altair as alt
import streamlit as st

from app_utils import load_app_data, monthly_volume, workout_date_bounds, ytd_volume

st.title("Monthly View")

data = load_app_data()
if not data:
    st.stop()

_, _, df = data
_, last_date = workout_date_bounds(df)

if last_date is not None:
    st.metric(f"YTD Volume ({last_date.year})", f"{ytd_volume(df, last_date.year):.0f} kg")

monthly_df = monthly_volume(df)
if monthly_df.empty:
    st.caption("No monthly volume data available.")
else:
    chart = (
        alt.Chart(monthly_df)
        .mark_bar()
        .encode(
            x=alt.X("month:N", title="Month"),
            y=alt.Y("volume:Q", title="Volume (kg·reps)"),
            tooltip=["month:N", "volume:Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(monthly_df, use_container_width=True)
