#!/usr/bin/env python3
"""Streamlit app for Gym Stats Insights."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

DATA_PATH = Path("data/mock_history.json")


@st.cache_data(show_spinner=False)
def load_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def history_dataframe(history: list[dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df["reps"] = pd.to_numeric(df["reps"], errors="coerce").fillna(0).astype(int)
    df["setNumber"] = pd.to_numeric(df["setNumber"], errors="coerce").fillna(0).astype(int)
    return df


def compute_summary(df: pd.DataFrame) -> dict:
    workouts = df[df["type"] == "workout"]
    total_sessions = workouts["date"].nunique()
    total_sets = len(workouts)
    total_reps = int(workouts["reps"].sum())
    total_volume = float((workouts["weight"] * workouts["reps"]).sum())
    return {
        "total_sessions": total_sessions,
        "total_sets": total_sets,
        "total_reps": total_reps,
        "total_volume": total_volume,
    }


def per_day_volume(df: pd.DataFrame) -> pd.DataFrame:
    workouts = df[df["type"] == "workout"].copy()
    workouts["volume"] = workouts["weight"] * workouts["reps"]
    return (
        workouts.groupby("date", as_index=False)["volume"].sum()
        .sort_values("date")
        .reset_index(drop=True)
    )


def exercise_progression(df: pd.DataFrame) -> pd.DataFrame:
    workouts = df[df["type"] == "workout"].copy()
    return workouts.sort_values("timestamp")


def last_session_per_exercise(df: pd.DataFrame) -> pd.DataFrame:
    workouts = df[df["type"] == "workout"].copy()
    if workouts.empty:
        return pd.DataFrame()

    last_dates = workouts.groupby("exercise")["date"].max().rename("last_date")
    merged = workouts.merge(last_dates, left_on="exercise", right_index=True)
    last_day = merged[merged["date"] == merged["last_date"]]
    agg = last_day.groupby("exercise").agg(
        last_date=("date", "max"),
        top_weight=("weight", "max"),
        total_sets=("exercise", "count"),
        total_reps=("reps", "sum"),
    )
    return agg.sort_index().reset_index()


def template_table(template: list[dict]) -> pd.DataFrame:
    if not template:
        return pd.DataFrame()
    df = pd.DataFrame(template)
    return df[["name", "defaultWeight", "defaultReps"]].rename(
        columns={
            "name": "exercise",
            "defaultWeight": "target_weight",
            "defaultReps": "target_reps",
        }
    )


def see_by_reps(reps: int, base: float, slope: float, max_reps: int, max_see: float) -> float:
    if reps <= 3:
        return base
    if reps <= max_reps:
        return base + slope * (reps - 3)
    return max_see


def estimate_1rm(row: pd.Series) -> dict:
    exercise = str(row.get("exercise", "")).lower()
    weight = float(row.get("weight", 0.0))
    reps = int(row.get("reps", 0))

    if "deadlift" in exercise:
        estimate = 1.04 * weight * (1 + reps / 30)
        see = see_by_reps(reps, 3.5, 1.5, 8, 11.0)
    elif "squat" in exercise:
        estimate = weight * (reps**0.10)
        see = see_by_reps(reps, 3.0, 1.5, 8, 9.0)
    elif "bench" in exercise:
        estimate = weight * (reps**0.10)
        see = see_by_reps(reps, 2.0, 0.7, 10, 7.5)
    elif "overhead" in exercise or "press" in exercise:
        estimate = 1.04 * weight * (1 + reps / 30)
        see = see_by_reps(reps, 2.5, 0.5, 10, 6.5)
    elif "row" in exercise:
        estimate = 0.93 * weight * (reps**0.10)
        see = see_by_reps(reps, 2.5, 1.0, 10, 7.0)
    elif "pull up" in exercise or "pull-up" in exercise:
        estimate = weight * (1 + reps / 30)
        see = see_by_reps(reps, 2.0, 1.0, 10, 6.0)
    else:
        estimate = weight * (1 + reps / 30)
        see = 0.0

    ci_delta = 1.96 * see
    return {
        "estimate": float(estimate),
        "ci_low": float(estimate - ci_delta),
        "ci_high": float(estimate + ci_delta),
        "see": float(see),
    }


def first_set_1rm(df: pd.DataFrame) -> pd.DataFrame:
    workouts = df[(df["type"] == "workout") & (df["setNumber"] == 1)].copy()
    if workouts.empty:
        return pd.DataFrame()
    estimates = workouts.apply(estimate_1rm, axis=1, result_type="expand")
    merged = pd.concat([workouts, estimates], axis=1)
    return merged.sort_values("timestamp")


def exercise_snapshot(df: pd.DataFrame, exercise: str) -> dict:
    data = df[(df["type"] == "workout") & (df["exercise"] == exercise)].copy()
    if data.empty:
        return {
            "total_sets": 0,
            "total_reps": 0,
            "total_volume": 0.0,
            "best_weight": 0.0,
            "last_date": None,
        }
    data["volume"] = data["weight"] * data["reps"]
    return {
        "total_sets": len(data),
        "total_reps": int(data["reps"].sum()),
        "total_volume": float(data["volume"].sum()),
        "best_weight": float(data["weight"].max()),
        "last_date": data["date"].max(),
    }


def main() -> None:
    st.set_page_config(page_title="Gym Stats Insights", layout="wide")
    st.title("Gym Stats Insights")

    payload = load_payload(DATA_PATH)
    if not payload:
        st.warning(
            "Mock data not found. Generate it with `python3 scripts/fetch_history.py` "
            "or place JSON at data/mock_history.json."
        )
        return

    history = payload.get("history", [])
    template = payload.get("template", [])
    df = history_dataframe(history)

    if df.empty:
        st.info("No history records found in the mock data.")
        return

    summary = compute_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions", summary["total_sessions"])
    col2.metric("Total Sets", summary["total_sets"])
    col3.metric("Total Reps", summary["total_reps"])
    col4.metric("Total Volume", f"{summary['total_volume']:.0f} kg")

    st.subheader("Session Volume")
    volume_df = per_day_volume(df)
    volume_chart = (
        alt.Chart(volume_df)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("volume:Q", title="Volume (kg·reps)"),
            tooltip=["date:T", "volume:Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(volume_chart, use_container_width=True)

    st.subheader("Exercise Progression")
    exercises = sorted(df[df["type"] == "workout"]["exercise"].unique())
    selected = st.multiselect("Exercises", exercises, default=exercises)
    prog_df = exercise_progression(df)
    prog_df = prog_df[prog_df["exercise"].isin(selected)]

    if not prog_df.empty:
        base = alt.Chart(prog_df).encode(
            x=alt.X("date:T", title="Date"),
            color=alt.Color("exercise:N", title="Exercise"),
            tooltip=["date:T", "exercise:N", "weight:Q", "reps:Q", "setNumber:Q"],
        )

        weight_chart = (
            base.mark_line(point=True)
            .encode(y=alt.Y("weight:Q", title="Weight (kg)", scale=alt.Scale(zero=False)))
            .properties(height=240)
        )
        reps_chart = (
            base.mark_line(point=True)
            .encode(y=alt.Y("reps:Q", title="Reps", scale=alt.Scale(zero=False)))
            .properties(height=240)
        )
        st.altair_chart(weight_chart, use_container_width=True)
        st.altair_chart(reps_chart, use_container_width=True)
    else:
        st.info("Select at least one exercise to see progression.")

    st.subheader("Per-Exercise View")
    if exercises:
        first_set_all = first_set_1rm(df)
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

    st.subheader("Last Session per Exercise")
    st.dataframe(last_session_per_exercise(df), use_container_width=True)

    st.subheader("Template Targets")
    st.dataframe(template_table(template), use_container_width=True)

    st.subheader("Sick Days")
    sick = df[df["type"] == "sick"]
    if sick.empty:
        st.caption("No sick days logged.")
    else:
        sick_dates = sick["date"].drop_duplicates().sort_values().tolist()
        st.write(", ".join(sick_dates))


if __name__ == "__main__":
    main()
