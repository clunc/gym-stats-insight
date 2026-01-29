"""Per-exercise detail view."""

from __future__ import annotations

import altair as alt
import streamlit as st

from app_utils import (
    exercise_progression,
    exercise_snapshot,
    first_set_1rm,
    load_app_data,
)

st.title("Per-Exercise View")

data = load_app_data()
if not data:
    st.stop()

_, _, df = data
exercises = sorted(df[df["type"] == "workout"]["exercise"].unique())

if exercises:
    first_set_all = first_set_1rm(df)
    prog_df = exercise_progression(df)
    tabs = st.tabs(exercises)
    for exercise, tab in zip(exercises, tabs):
        with tab:
            ex_df = prog_df[prog_df["exercise"] == exercise]
            snapshot = exercise_snapshot(df, exercise)

            stat1, stat2, stat3, stat4 = st.columns(4)
            stat1.metric("Total Sets", snapshot["total_sets"])
            stat2.metric("Total Reps", snapshot["total_reps"])
            stat3.metric("Total Volume", f"{snapshot['total_volume']:.0f} kg")
            stat4.metric("Best Weight", f"{snapshot['best_weight']:.1f} kg")

            st.caption(f"Last session: {snapshot['last_date'] or '—'}")

            if ex_df.empty:
                st.info("No sessions recorded for this exercise.")
                continue

            base = alt.Chart(ex_df).encode(
                x=alt.X("date:T", title="Date"),
                tooltip=["date:T", "weight:Q", "reps:Q", "setNumber:Q"],
            )

            weight_chart = (
                base.mark_line(point=True)
                .encode(y=alt.Y("weight:Q", title="Weight (kg)", scale=alt.Scale(zero=False)))
                .properties(height=220)
            )
            reps_chart = (
                base.mark_line(point=True)
                .encode(y=alt.Y("reps:Q", title="Reps", scale=alt.Scale(zero=False)))
                .properties(height=220)
            )
            st.altair_chart(weight_chart, use_container_width=True)
            st.altair_chart(reps_chart, use_container_width=True)

            st.markdown("**1RM Estimate Over Time (First Set)**")
            first_set = first_set_all[first_set_all["exercise"] == exercise]
            if first_set.empty:
                st.caption("No first-set data for 1RM estimates.")
            else:
                ci_band = (
                    alt.Chart(first_set)
                    .mark_area(opacity=0.2)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("ci_low:Q", title="1RM (kg)", scale=alt.Scale(zero=False)),
                        y2="ci_high:Q",
                    )
                    .properties(height=220)
                )
                est_line = (
                    alt.Chart(first_set)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("estimate:Q", title="1RM (kg)", scale=alt.Scale(zero=False)),
                        tooltip=["date:T", "estimate:Q", "ci_low:Q", "ci_high:Q"],
                    )
                    .properties(height=220)
                )
                st.altair_chart(ci_band + est_line, use_container_width=True)

            st.dataframe(
                ex_df[["date", "weight", "reps", "setNumber"]],
                use_container_width=True,
            )
else:
    st.info("No exercises available for per-exercise tabs.")
