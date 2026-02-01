"""Per-exercise detail view."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app_utils import (
    e1rm_formula_for_exercise,
    e1rm_latest_delta,
    e1rm_pr_flags,
    e1rm_series,
    exercise_snapshot,
    load_app_data,
    progress_to_next_increase,
)

st.title("Per-Exercise View")

data = load_app_data()
if not data:
    st.stop()

_, _, df = data
exercises = sorted(df[df["type"] == "workout"]["exercise"].unique())

if not exercises:
    st.info("No exercises available for per-exercise view.")
    st.stop()


def render_exercise_view(selected_exercise: str) -> None:
    time_range = st.radio(
        "Time Range",
        ("All", "1Y", "YTD", "6M", "3M", "1M", "1W"),
        horizontal=True,
        key=f"time_range_{selected_exercise}",
    )
    full_series = e1rm_series(df, selected_exercise)
    series = full_series
    if series.empty:
        st.info("No sessions recorded for this exercise.")
        return

    max_date = series["timestamp"].max()
    cutoff = None
    if time_range != "All":
        if time_range == "YTD":
            cutoff = pd.Timestamp(year=max_date.year, month=1, day=1, tz=max_date.tz)
            series = series[series["timestamp"] >= cutoff]
        elif time_range == "1W":
            cutoff = max_date - pd.DateOffset(weeks=1)
            series = series[series["timestamp"] >= cutoff]
        else:
            months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}[time_range]
            cutoff = max_date - pd.DateOffset(months=months)
            series = series[series["timestamp"] >= cutoff]

    series = e1rm_pr_flags(series)
    full_series = e1rm_pr_flags(full_series)
    latest = e1rm_latest_delta(series, full_series)

    st.subheader("Overview Stats")
    snapshot = exercise_snapshot(df, selected_exercise)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sets", snapshot["total_sets"])
    col2.metric("Total Reps", snapshot["total_reps"])
    col3.metric("Total Volume", f"{snapshot['total_volume']:.0f} kg")
    col4.metric("Best Weight", f"{snapshot['best_weight']:.1f} kg")

    st.subheader("Progress to Next Increase")
    progress = progress_to_next_increase(df, selected_exercise)
    if progress["current_weight"] is None:
        st.caption("No recent session to evaluate progression.")
    else:
        if progress["ready"]:
            if progress["next_weight"] is None:
                st.success(f"\U0001F3AF READY: {selected_exercise} is ready to increase.")
            else:
                st.success(
                    f"\U0001F3AF READY: {selected_exercise} {progress['current_weight']:.1f} kg "
                    f"\u2192 {progress['next_weight']:.1f} kg (+{progress['increase_pct']:.0f}%)"
                )
        else:
            st.info(
                f"{progress['sets_at_cap']}/3 sets at cap ({progress['cap']} reps) "
                f"\u2192 {progress['progress_pct']:.0f}% to next weight increase"
            )
            if progress["next_weight"] is not None:
                st.caption(
                    f"Target: {progress['current_weight']:.1f} kg \u2192 "
                    f"{progress['next_weight']:.1f} kg (+{progress['increase_pct']:.0f}%)"
                )

    st.subheader("Current 1ERM")
    if latest["current"] is None:
        st.caption("No 1ERM estimates available.")
    else:
        delta = latest["delta_pct"]
        delta_abs = latest["delta_abs"]
        if delta is None or delta_abs is None:
            delta_label = "—"
        else:
            delta_label = f"{delta_abs:+.1f} kg ({delta:+.1f}%)"
        st.metric(
            "1ERM (with CI)",
            f"{latest['current']:.1f} kg ± {abs(latest['current'] - latest['ci_low']):.1f} kg",
            delta_label,
        )
        formula_name, formula = e1rm_formula_for_exercise(selected_exercise)
        st.caption(f"Formula (first set, {formula_name}): {formula}")

    st.subheader("1ERM Progression")
    ci_band = (
        alt.Chart(series)
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("ci_low:Q", title="1ERM (kg)", scale=alt.Scale(zero=False)),
            y2="ci_high:Q",
        )
        .properties(height=240)
    )
    est_line = (
        alt.Chart(series)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("estimate:Q", title="1ERM (kg)", scale=alt.Scale(zero=False)),
            tooltip=["date:T", "estimate:Q", "ci_low:Q", "ci_high:Q"],
        )
        .properties(height=240)
    )
    pr_points = (
        alt.Chart(series[series["pr"]])
        .mark_point(color="gold", size=80)
        .encode(x="date:T", y="estimate:Q", tooltip=["date:T", "estimate:Q"])
    )
    st.altair_chart(ci_band + est_line + pr_points, use_container_width=True)

    st.subheader("Session History")
    ex_df = df[(df["type"] == "workout") & (df["exercise"] == selected_exercise)].copy()
    if cutoff is not None:
        ex_df = ex_df[ex_df["timestamp"] >= cutoff]
    if ex_df.empty:
        st.caption("No session history.")
    else:
        weight_by_set = (
            ex_df.pivot_table(index="date", columns="setNumber", values="weight", aggfunc="max")
            .rename(columns={1: "Set 1", 2: "Set 2", 3: "Set 3"})
            .reset_index()
        )
        weight_long = weight_by_set.melt("date", var_name="set", value_name="weight").dropna()
        weight_chart = (
            alt.Chart(weight_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("weight:Q", title="Weight (kg)", scale=alt.Scale(zero=False)),
                color=alt.Color("set:N", title="Set"),
                tooltip=["date:T", "set:N", "weight:Q"],
            )
            .properties(height=220)
        )
        st.altair_chart(weight_chart, use_container_width=True)

        reps_by_set = (
            ex_df.pivot_table(index="date", columns="setNumber", values="reps", aggfunc="max")
            .rename(columns={1: "Set 1", 2: "Set 2", 3: "Set 3"})
            .reset_index()
        )
        reps_long = reps_by_set.melt("date", var_name="set", value_name="reps").dropna()
        reps_chart = (
            alt.Chart(reps_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("reps:Q", title="Reps", scale=alt.Scale(zero=False)),
                color=alt.Color("set:N", title="Set"),
                tooltip=["date:T", "set:N", "reps:Q"],
            )
            .properties(height=220)
        )
        st.altair_chart(reps_chart, use_container_width=True)

        e1rm = e1rm_series(df, selected_exercise)
        e1rm_by_date = e1rm.groupby("date", as_index=False)["estimate"].max()
        sets = (
            ex_df.pivot_table(index="date", columns="setNumber", values="reps", aggfunc="max")
            .rename(columns={1: "reps_set1", 2: "reps_set2", 3: "reps_set3"})
            .reset_index()
        )
        weights = ex_df.groupby("date", as_index=False)["weight"].max().rename(columns={"weight": "weight"})
        volume = ex_df.groupby("date", as_index=False).apply(
            lambda frame: float((frame["weight"] * frame["reps"]).sum())
        ).rename(columns={0: "volume"})
        table = weights.merge(sets, on="date", how="left").merge(e1rm_by_date, on="date", how="left").merge(
            volume, on="date", how="left"
        )
        table = table.sort_values("date", ascending=True)
        table["pr"] = table["estimate"].cummax() == table["estimate"]
        table["pr"] = table["pr"].map(lambda val: "PR" if val else "")
        table = table.sort_values("date", ascending=False)
        st.dataframe(table, use_container_width=True)


tabs = st.tabs(exercises)
for tab, exercise in zip(tabs, exercises):
    with tab:
        render_exercise_view(exercise)
