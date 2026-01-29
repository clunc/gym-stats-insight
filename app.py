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
            .encode(y=alt.Y("weight:Q", title="Weight (kg)"))
            .properties(height=240)
        )
        reps_chart = (
            base.mark_line(point=True)
            .encode(y=alt.Y("reps:Q", title="Reps"))
            .properties(height=240)
        )
        st.altair_chart(weight_chart, use_container_width=True)
        st.altair_chart(reps_chart, use_container_width=True)
    else:
        st.info("Select at least one exercise to see progression.")

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
