"""Shared utilities for Gym Stats Insights Streamlit pages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
import streamlit as st

DATA_PATH = Path("data/mock_history.json")
ENV_PATH = Path(".env")


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_payload(path: Path) -> dict:
    load_env_file()
    url = os.getenv("GYM_HISTORY_URL", "").strip()
    if url:
        try:
            with urlopen(url, timeout=10) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)
        except (HTTPError, URLError, json.JSONDecodeError):
            # Fall back to local file if remote source is unavailable.
            pass
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


def volume_trend(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    workouts = df[df["type"] == "workout"].copy()
    if workouts.empty:
        return pd.DataFrame(columns=["date", "volume"])
    workouts["volume"] = workouts["weight"] * workouts["reps"]
    if freq == "D":
        daily = workouts.groupby("date", as_index=False)["volume"].sum().sort_values("date")
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        return daily.reset_index(drop=True)
    if freq == "W":
        workouts["week_start"] = workouts["timestamp"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
        weekly = (
            workouts.groupby("week_start", as_index=False)["volume"].sum().sort_values("week_start")
        )
        weekly = weekly.rename(columns={"week_start": "date"})
        return weekly.reset_index(drop=True)
    raise ValueError("freq must be 'D' or 'W'")


def weekly_volume_metrics(df: pd.DataFrame) -> dict:
    workouts = df[df["type"] == "workout"].copy()
    if workouts.empty:
        return {"this_week": 0.0, "last_week": 0.0, "pct_change": None, "avg_4w": None}
    workouts["volume"] = workouts["weight"] * workouts["reps"]
    workouts["week_start"] = workouts["timestamp"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    weekly = workouts.groupby("week_start", as_index=False)["volume"].sum()
    latest_week = weekly["week_start"].max()
    this_week = float(weekly.loc[weekly["week_start"] == latest_week, "volume"].sum())
    last_week_start = latest_week - pd.Timedelta(days=7)
    last_week = float(weekly.loc[weekly["week_start"] == last_week_start, "volume"].sum())
    pct_change = None if last_week == 0 else (this_week - last_week) / last_week * 100
    weekly_sorted = weekly.sort_values("week_start")
    avg_4w = float(weekly_sorted["volume"].tail(4).mean()) if not weekly_sorted.empty else None
    return {"this_week": this_week, "last_week": last_week, "pct_change": pct_change, "avg_4w": avg_4w}


def weekly_streak_metrics(df: pd.DataFrame, min_sessions_per_week: int = 2) -> dict:
    data = df[df["type"].isin(["workout", "sick"])].copy()
    if data.empty:
        return {
            "current_streak_sessions": 0,
            "longest_streak_sessions": 0,
            "current_streak_weeks": 0,
            "longest_streak_weeks": 0,
        }

    data["week_start"] = data["timestamp"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    weekly = (
        data.groupby("week_start")["date"]
        .nunique()
        .rename("sessions_per_week")
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    week_min = weekly["week_start"].min()
    week_max = weekly["week_start"].max()
    full_weeks = pd.DataFrame({"week_start": pd.date_range(start=week_min, end=week_max, freq="W-MON")})
    weekly = full_weeks.merge(weekly, on="week_start", how="left").fillna({"sessions_per_week": 0})
    weekly["sessions_per_week"] = weekly["sessions_per_week"].astype(int)

    latest_week = weekly["week_start"].max()
    weekly["is_completed_week"] = weekly["week_start"] < latest_week
    weekly["is_qualified_or_open"] = (~weekly["is_completed_week"]) | (
        weekly["sessions_per_week"] >= min_sessions_per_week
    )

    # Longest run: completed weeks must qualify; latest week is non-breaking/in-progress.
    longest_weeks = 0
    longest_sessions = 0
    run_weeks = 0
    run_sessions = 0
    for row in weekly.itertuples(index=False):
        if bool(row.is_qualified_or_open):
            run_weeks += 1
            run_sessions += int(row.sessions_per_week)
            if run_sessions > longest_sessions:
                longest_sessions = run_sessions
                longest_weeks = run_weeks
        else:
            run_weeks = 0
            run_sessions = 0

    # Current run from latest week backward; latest week never breaks streak.
    current_weeks = 0
    current_sessions = 0
    rows_rev = weekly.sort_values("week_start", ascending=False).itertuples(index=False)
    for row in rows_rev:
        if bool(row.is_qualified_or_open):
            current_weeks += 1
            current_sessions += int(row.sessions_per_week)
        else:
            break

    return {
        "current_streak_sessions": current_sessions,
        "longest_streak_sessions": longest_sessions,
        "current_streak_weeks": current_weeks,
        "longest_streak_weeks": longest_weeks,
    }


def rep_cap_for_exercise(exercise: str) -> int:
    name = exercise.lower()
    if "deadlift" in name or "squat" in name or "pull up" in name or "pull-up" in name:
        return 8
    return 10


def lower_rep_cap_for_exercise(exercise: str) -> int:
    name = exercise.lower()
    if "deadlift" in name or "squat" in name or "pull up" in name or "pull-up" in name:
        return 4
    return 6


def round_weight(value: float, step: float = 0.5) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def progress_to_next_increase(
    df: pd.DataFrame, exercise: str, increase_pct: float = 0.03, step: float = 0.5
) -> dict:
    session = latest_session(df, exercise)
    cap = rep_cap_for_exercise(exercise)
    sets_at_cap = 0
    for set_num in (1, 2, 3):
        row = session[session["setNumber"] == set_num]
        reps = int(row.iloc[0]["reps"]) if not row.empty else 0
        if reps >= cap:
            sets_at_cap += 1

    current_weight = None
    last_date = None
    if not session.empty:
        current_weight = float(session["weight"].max())
        last_date = session["date"].max()

    ready = sets_at_cap == 3
    progress_pct = sets_at_cap / 3 * 100
    next_weight = None
    if current_weight is not None and current_weight > 0:
        next_weight = round_weight(current_weight * (1 + increase_pct), step=step)

    return {
        "exercise": exercise,
        "cap": cap,
        "sets_at_cap": sets_at_cap,
        "progress_pct": progress_pct,
        "ready": ready,
        "current_weight": current_weight,
        "next_weight": next_weight,
        "last_date": last_date,
        "increase_pct": increase_pct * 100,
    }


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


def absolute_progression_kg(df: pd.DataFrame) -> pd.DataFrame:
    workouts = df[df["type"] == "workout"].copy()
    if workouts.empty:
        return pd.DataFrame(columns=["exercise", "progress_kg", "start_weight", "latest_weight"])
    per_session = (
        workouts.groupby(["exercise", "date"], as_index=False)["weight"].max()
        .sort_values(["exercise", "date"])
    )
    first = per_session.groupby("exercise", as_index=False).first()
    last = per_session.groupby("exercise", as_index=False).last()
    merged = first.merge(last, on="exercise", suffixes=("_start", "_latest"))
    merged["progress_kg"] = merged["weight_latest"] - merged["weight_start"]
    return merged.rename(
        columns={
            "weight_start": "start_weight",
            "weight_latest": "latest_weight",
        }
    )[
        ["exercise", "progress_kg", "start_weight", "latest_weight"]
    ].sort_values("exercise")


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


def latest_session(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    ex_df = df[(df["type"] == "workout") & (df["exercise"] == exercise)].copy()
    if ex_df.empty:
        return pd.DataFrame()
    latest_date = ex_df["date"].max()
    return ex_df[ex_df["date"] == latest_date].copy()


def session_summary(session: pd.DataFrame) -> dict:
    if session.empty:
        return {
            "date": None,
            "volume": 0.0,
            "weight_pr": 0.0,
            "sets": [],
        }
    date = session["date"].max()
    volume = float((session["weight"] * session["reps"]).sum())
    weight_pr = float(session["weight"].max())
    sets = []
    for set_num in (1, 2, 3):
        row = session[session["setNumber"] == set_num]
        if row.empty:
            sets.append({"set": set_num, "weight": None, "reps": None})
        else:
            sets.append(
                {
                    "set": set_num,
                    "weight": float(row.iloc[0]["weight"]),
                    "reps": int(row.iloc[0]["reps"]),
                }
            )
    return {"date": date, "volume": volume, "weight_pr": weight_pr, "sets": sets}


def e1rm_series(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    first_set = first_set_1rm(df)
    if first_set.empty:
        return pd.DataFrame()
    series = first_set[first_set["exercise"] == exercise].copy()
    return series.sort_values("timestamp")


def e1rm_latest_delta(series: pd.DataFrame, baseline_series: pd.DataFrame | None = None) -> dict:
    if series.empty:
        return {"current": None, "ci_low": None, "ci_high": None, "delta_pct": None, "delta_abs": None}
    current = series.iloc[-1]
    if baseline_series is None:
        baseline_series = series
    baseline = baseline_series.iloc[0] if not baseline_series.empty else None
    delta_pct = None
    delta_abs = None
    if baseline is not None and float(baseline["estimate"]) != 0:
        delta_abs = float(current["estimate"]) - float(baseline["estimate"])
        delta_pct = (float(current["estimate"]) - float(baseline["estimate"])) / float(baseline["estimate"]) * 100
    return {
        "current": float(current["estimate"]),
        "ci_low": float(current["ci_low"]),
        "ci_high": float(current["ci_high"]),
        "delta_pct": delta_pct,
        "delta_abs": delta_abs,
        "date": current["date"],
    }


def e1rm_formula_for_exercise(exercise: str) -> tuple[str, str]:
    name = exercise.lower()
    if "deadlift" in name:
        return ("Epley (scaled)", "Estimate = 1.04 x weight x (1 + reps/30)")
    if "squat" in name:
        return ("Lombardi", "Estimate = weight x reps^0.10")
    if "bench" in name:
        return ("Lombardi", "Estimate = weight x reps^0.10")
    if "overhead" in name or "press" in name:
        return ("Epley (scaled)", "Estimate = 1.04 x weight x (1 + reps/30)")
    if "row" in name:
        return ("Lombardi (scaled)", "Estimate = 0.93 x weight x reps^0.10")
    if "pull up" in name or "pull-up" in name:
        return ("Epley", "Estimate = weight x (1 + reps/30)")
    return ("Epley", "Estimate = weight x (1 + reps/30)")


def e1rm_pr_flags(series: pd.DataFrame) -> pd.DataFrame:
    if series.empty:
        return series
    series = series.copy()
    series["pr"] = series["estimate"].cummax() == series["estimate"]
    return series


def muscle_group_for_exercise(exercise: str) -> str:
    name = exercise.lower()
    if "bench" in name or "chest" in name:
        return "Chest"
    if "squat" in name or "quad" in name:
        return "Quads"
    if "deadlift" in name or "hamstring" in name:
        return "Hamstrings"
    if "overhead" in name or "press" in name:
        return "Shoulders"
    if "row" in name or "pull" in name or "lat" in name:
        return "Back"
    if "curl" in name:
        return "Biceps"
    if "tricep" in name or "extension" in name:
        return "Triceps"
    return "Other"


def weekly_sets_by_muscle(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["muscle", "sets"])
    df = df.copy()
    df["muscle"] = df["exercise"].apply(muscle_group_for_exercise)
    return df.groupby("muscle", as_index=False).size().rename(columns={"size": "sets"})


def weekly_sessions_by_muscle(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["muscle", "sessions"])
    df = df.copy()
    df["muscle"] = df["exercise"].apply(muscle_group_for_exercise)
    sessions = df.groupby(["muscle", "date"], as_index=False).size()
    return sessions.groupby("muscle", as_index=False).size().rename(columns={"size": "sessions"})


def weekly_e1rm_prs(df: pd.DataFrame, start_str: str, end_str: str) -> list[str]:
    if df.empty:
        return []
    first_set = first_set_1rm(df)
    if first_set.empty:
        return []
    first_set = first_set.sort_values("date")
    all_time = first_set.groupby("exercise", as_index=False)["estimate"].max()
    week = first_set[(first_set["date"] >= start_str) & (first_set["date"] <= end_str)]
    if week.empty:
        return []
    week_max = week.groupby("exercise", as_index=False)["estimate"].max()
    merged = week_max.merge(all_time, on="exercise", suffixes=("_week", "_all"))
    return merged[merged["estimate_week"] >= merged["estimate_all"]]["exercise"].tolist()


def exercise_status_by_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["exercise", "status"])
    first_set = first_set_1rm(df)
    if first_set.empty:
        return pd.DataFrame(columns=["exercise", "status"])
    first_set = first_set.copy()
    first_set["week_start"] = first_set["timestamp"].dt.to_period("W").apply(lambda p: p.start_time.date())
    weekly = first_set.groupby(["exercise", "week_start"], as_index=False)["estimate"].max()
    statuses = []
    for exercise, group in weekly.groupby("exercise"):
        group = group.sort_values("week_start")
        if len(group) < 2:
            statuses.append({"exercise": exercise, "status": "Yellow"})
            continue
        latest = group.iloc[-1]["estimate"]
        prev = group.iloc[-2]["estimate"]
        if latest > prev:
            status = "Green"
        elif latest < prev:
            status = "Red"
        else:
            status = "Yellow"
        statuses.append({"exercise": exercise, "status": status})
    return pd.DataFrame(statuses)


def rep_shortfall_percent(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    shortfalls = []
    for exercise, ex_df in df.groupby("exercise"):
        cap = rep_cap_for_exercise(exercise)
        sets = ex_df[ex_df["setNumber"].isin([1, 2])]
        if sets.empty:
            continue
        target = cap * len(sets)
        actual = int(sets["reps"].clip(upper=cap).sum())
        if target > 0:
            shortfalls.append((target - actual) / target)
    if not shortfalls:
        return 0.0
    return float(sum(shortfalls) / len(shortfalls) * 100)
