"""Shared utilities for Gym Stats Insights Streamlit pages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

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


def load_app_data() -> tuple[list[dict], list[dict], pd.DataFrame] | None:
    payload = load_payload(DATA_PATH)
    if not payload:
        st.warning(
            "Mock data not found. Generate it with `python3 scripts/fetch_history.py` "
            "or place JSON at data/mock_history.json."
        )
        return None

    history = payload.get("history", [])
    template = payload.get("template", [])
    df = history_dataframe(history)

    if df.empty:
        st.info("No history records found in the mock data.")
        return None

    return history, template, df




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


def volume_over_time(df: pd.DataFrame, weekly_threshold_days: int = 90) -> tuple[pd.DataFrame, str]:
    volume_df = per_day_volume(df)
    if volume_df.empty:
        return volume_df, "day"
    dates = pd.to_datetime(volume_df["date"], errors="coerce")
    span_days = (dates.max() - dates.min()).days if dates.notna().any() else 0
    if span_days > weekly_threshold_days:
        volume_df = volume_df.assign(week_start=dates.dt.to_period("W").apply(lambda p: p.start_time))
        weekly = (
            volume_df.groupby("week_start", as_index=False)["volume"].sum()
            .sort_values("week_start")
            .reset_index(drop=True)
        )
        weekly["date"] = weekly["week_start"].dt.date.astype(str)
        return weekly[["date", "volume"]], "week"
    return volume_df, "day"


def top_exercises_by_volume(df: pd.DataFrame, limit: int = 5) -> list[str]:
    workouts = df[df["type"] == "workout"].copy()
    if workouts.empty:
        return []
    workouts["volume"] = workouts["weight"] * workouts["reps"]
    ranked = (
        workouts.groupby("exercise", as_index=True)["volume"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return ranked[:limit]


def workout_date_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    workouts = df[df["type"] == "workout"]
    if workouts.empty:
        return None, None
    return workouts["timestamp"].min(), workouts["timestamp"].max()


def weekly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["exercise", "weekly_volume", "sets"])
    df = df.copy()
    df["volume"] = df["weight"] * df["reps"]
    summary = (
        df.groupby("exercise", as_index=False)
        .agg(weekly_volume=("volume", "sum"), sets=("exercise", "count"))
        .sort_values("weekly_volume", ascending=False)
    )
    return summary


def compound_exercises(exercises: list[str]) -> list[str]:
    keywords = ("deadlift", "squat", "bench", "overhead", "press", "row")
    compounds = [ex for ex in exercises if any(k in ex.lower() for k in keywords)]
    return sorted(compounds)


def weekly_volume_context(ex_df: pd.DataFrame) -> dict:
    if ex_df.empty:
        return {"weekly_volume": 0.0, "rolling_avg": 0.0, "trend": "→"}
    data = ex_df.copy()
    data["volume"] = data["weight"] * data["reps"]
    data["week_start"] = data["timestamp"].dt.to_period("W").apply(lambda p: p.start_time.date())
    weekly = data.groupby("week_start", as_index=False)["volume"].sum().sort_values("week_start")
    weekly["rolling_avg"] = weekly["volume"].rolling(3, min_periods=1).mean()
    last = weekly.iloc[-1]
    weekly_volume = float(last["volume"])
    rolling_avg = float(last["rolling_avg"])
    if rolling_avg == 0:
        trend = "→"
    elif weekly_volume > rolling_avg * 1.02:
        trend = "↑"
    elif weekly_volume < rolling_avg * 0.98:
        trend = "↓"
    else:
        trend = "→"
    return {"weekly_volume": weekly_volume, "rolling_avg": rolling_avg, "trend": trend}


def rep_cap_for_exercise(exercise: str) -> int:
    name = exercise.lower()
    if "deadlift" in name or "squat" in name or "pull up" in name or "pull-up" in name:
        return 8
    return 10


def progression_status(session: pd.DataFrame, cap: int) -> dict:
    missing = 0
    ready = True
    for set_num in (1, 2, 3):
        row = session[session["setNumber"] == set_num]
        reps = int(row.iloc[0]["reps"]) if not row.empty else 0
        if reps < cap:
            ready = False
            missing += cap - reps
    if session.empty:
        ready = False
        missing = cap * 3
    return {"ready": ready, "missing": missing}


def progression_event_dates(ex_df: pd.DataFrame, cap: int) -> list[str]:
    if ex_df.empty:
        return []
    events: list[str] = []
    for date in sorted(ex_df["date"].unique()):
        session = ex_df[ex_df["date"] == date]
        if progression_status(session, cap)["ready"]:
            events.append(date)
    return events


def progression_event_summary(df: pd.DataFrame, exercises: list[str]) -> tuple[int, pd.DataFrame]:
    rows = []
    total_events = 0
    for exercise in exercises:
        ex_df = df[(df["type"] == "workout") & (df["exercise"] == exercise)]
        cap = rep_cap_for_exercise(exercise)
        events = progression_event_dates(ex_df, cap)
        total_events += len(events)
        rows.append(
            {
                "exercise": exercise,
                "event_count": len(events),
                "last_event": max(events) if events else "—",
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["event_count", "exercise"], ascending=[False, True])
    return total_events, summary


def stagnation_flag_in_period(
    ex_df: pd.DataFrame, cap: int, start_date: str, end_date: str
) -> bool:
    if ex_df.empty:
        return False
    sessions: list[tuple[str, tuple[float, float, int, int]]] = []
    for date in sorted(ex_df["date"].unique()):
        session = ex_df[ex_df["date"] == date]
        set1 = session[session["setNumber"] == 1]
        set2 = session[session["setNumber"] == 2]
        if set1.empty or set2.empty:
            continue
        w1 = float(set1.iloc[0]["weight"])
        w2 = float(set2.iloc[0]["weight"])
        r1 = int(set1.iloc[0]["reps"])
        r2 = int(set2.iloc[0]["reps"])
        sessions.append((date, (w1, w2, min(r1, cap), min(r2, cap))))
    if len(sessions) < 3:
        return False
    last_three = sessions[-3:]
    if not all(start_date <= date <= end_date for date, _ in last_three):
        return False
    return last_three[0][1] == last_three[1][1] == last_three[2][1]


def monthly_volume(df: pd.DataFrame) -> pd.DataFrame:
    workouts = df[df["type"] == "workout"].copy()
    if workouts.empty:
        return pd.DataFrame(columns=["month", "volume"])
    workouts["volume"] = workouts["weight"] * workouts["reps"]
    workouts["month"] = workouts["timestamp"].dt.to_period("M").astype(str)
    monthly = (
        workouts.groupby("month", as_index=False)["volume"].sum()
        .sort_values("month")
        .reset_index(drop=True)
    )
    return monthly


def ytd_volume(df: pd.DataFrame, year: int) -> float:
    workouts = df[(df["type"] == "workout") & (df["timestamp"].dt.year == year)]
    if workouts.empty:
        return 0.0
    return float((workouts["weight"] * workouts["reps"]).sum())


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


def latest_1rm_average(df: pd.DataFrame) -> float | None:
    first_set = first_set_1rm(df)
    if first_set.empty:
        return None
    latest = (
        first_set.sort_values("date")
        .groupby("exercise", as_index=False)
        .tail(1)
    )
    if latest.empty:
        return None
    return float(latest["estimate"].mean())


def latest_1rm_table(df: pd.DataFrame) -> pd.DataFrame:
    first_set = first_set_1rm(df)
    if first_set.empty:
        return pd.DataFrame(columns=["exercise", "latest_1rm", "date"])
    latest = (
        first_set.sort_values("date")
        .groupby("exercise", as_index=False)
        .tail(1)
        .rename(columns={"estimate": "latest_1rm"})
    )
    return latest[["exercise", "latest_1rm", "date"]].sort_values("exercise")
