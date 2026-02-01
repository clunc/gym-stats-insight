"""Overview (All Time) page."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app_utils import (
    absolute_progression_kg,
    compute_summary,
    load_app_data,
    latest_1rm_table,
    progression_event_summary,
    workout_date_bounds,
)

st.title("Overview")

data = load_app_data()
if not data:
    st.stop()

_, template, df = data

workouts = df[df["type"] == "workout"]
all_workouts = workouts.copy()
if workouts.empty:
    st.info("No workout data available.")
    st.stop()

time_range = st.radio(
    "Time Range",
    ("All", "1Y", "YTD", "6M", "3M", "1M", "1W"),
    horizontal=True,
    key="overview_time_range",
)

max_date = workouts["timestamp"].max()
cutoff = None
if time_range != "All":
    if time_range == "YTD":
        cutoff = pd.Timestamp(year=max_date.year, month=1, day=1, tz=max_date.tz)
    elif time_range == "1W":
        cutoff = max_date - pd.DateOffset(weeks=1)
    else:
        months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}[time_range]
        cutoff = max_date - pd.DateOffset(months=months)

if cutoff is not None:
    df = df[df["timestamp"] >= cutoff].copy()
    workouts = df[df["type"] == "workout"]

first_date, last_date = workout_date_bounds(df)
if first_date is not None:
    if cutoff is not None:
        st.caption(f"Showing data since {first_date.date()}")
    else:
        st.caption(f"Since first logged workout on {first_date.date()}")

st.subheader("Workout Calendar")
daily_sessions = (
    workouts.groupby("date", as_index=False)
    .size()
    .rename(columns={"size": "sessions"})
)
if daily_sessions.empty:
    st.caption("No workouts to show in the selected time range.")
else:
    date_index = pd.date_range(
        start=pd.to_datetime(daily_sessions["date"]).min(),
        end=pd.to_datetime(daily_sessions["date"]).max(),
        freq="D",
    )
    calendar = pd.DataFrame({"date": date_index})
    calendar = calendar.merge(
        daily_sessions.assign(date=pd.to_datetime(daily_sessions["date"])),
        on="date",
        how="left",
    ).fillna({"sessions": 0})
    sick_dates = df[df["type"] == "sick"]["date"].drop_duplicates()
    calendar["is_sick"] = calendar["date"].isin(pd.to_datetime(sick_dates))
    streak_source = df[df["type"].isin(["workout", "sick"])]
    if time_range == "YTD" and cutoff is not None:
        week_start_cutoff = cutoff.to_period("W-SUN").start_time.date()
        streak_source = streak_source[streak_source["timestamp"].dt.date >= week_start_cutoff]

    workouts_with_week = streak_source.copy()
    workouts_with_week["week_start"] = workouts_with_week["timestamp"].dt.to_period("W-SUN").apply(
        lambda period: period.start_time
    )
    weekly_sessions = (
        workouts_with_week.groupby("week_start")["date"]
        .nunique()
        .rename("sessions_per_week")
        .reset_index()
    )
    weekly_sessions["is_streak"] = weekly_sessions["sessions_per_week"] >= 2
    calendar["week_start"] = calendar["date"].dt.to_period("W-SUN").apply(lambda period: period.start_time)
    calendar = calendar.merge(weekly_sessions[["week_start", "is_streak"]], on="week_start", how="left")
    calendar["is_streak"] = calendar["is_streak"].fillna(False)
    calendar["weekday_idx"] = calendar["date"].dt.isocalendar().day.astype(int)
    calendar["month"] = calendar["date"].dt.to_period("M").dt.to_timestamp()
    calendar["month_start"] = calendar["month"]
    calendar["month_start_weekday"] = calendar["month_start"].dt.isocalendar().day.astype(int) - 1
    calendar["day_of_month"] = calendar["date"].dt.day
    calendar["week_of_month"] = (
        (calendar["day_of_month"] - 1 + calendar["month_start_weekday"]) // 7
    ).astype(int)
    calendar["day"] = calendar["date"].dt.day.astype(str)
    calendar["is_header"] = False
    calendar["day_label"] = calendar["day"]

    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    header_rows = []
    for month in sorted(calendar["month"].unique()):
        for idx, label in enumerate(weekdays, start=1):
            header_rows.append(
                {
                    "date": month,
                    "sessions": 0,
                    "weekday_idx": idx,
                    "month": month,
                    "month_start": month,
                    "month_start_weekday": None,
                    "day_of_month": None,
                    "week_of_month": -1,
                    "day": "",
                    "is_header": True,
                    "day_label": label,
                    "is_streak": False,
                    "week_start": month,
                    "is_sick": False,
                }
            )
    header_df = pd.DataFrame(header_rows)
    calendar_with_header = pd.concat([calendar, header_df], ignore_index=True)

    base = alt.Chart(calendar_with_header).encode(
        x=alt.X(
            "weekday_idx:O",
            title=None,
            scale=alt.Scale(domain=list(range(1, 8))),
            axis=alt.Axis(labelOpacity=0, ticks=False),
        ),
        y=alt.Y(
            "week_of_month:O",
            title=None,
            axis=alt.Axis(labelOpacity=0),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("sessions:Q", title="Sessions"),
        ],
    ).properties(height=140)

    header_rect = base.transform_filter("datum.is_header === true").mark_rect().encode(
        color=alt.value("transparent")
    )
    week_rect = base.transform_filter("datum.is_header === false").mark_rect().encode(
        color=alt.condition(
            alt.datum.is_streak,
            alt.value("#C8E6C9"),
            alt.value("#FFCDD2"),
        ),
        opacity=alt.value(1),
    )
    sick_rect = base.transform_filter("datum.is_sick === true").mark_rect().encode(
        color=alt.value("#d97706"),
        opacity=alt.value(1),
    )
    day_rect = base.transform_filter("datum.sessions > 0").mark_rect().encode(
        color=alt.value("#2e7d32"),
        opacity=alt.value(1),
    )
    labels = base.mark_text(fontSize=9, dy=1).encode(
        text="day_label:N",
        color=alt.condition(
            alt.datum.is_header,
            alt.value("#ffffff"),
            alt.value("#111827"),
        ),
    )
    layered = alt.layer(header_rect, week_rect, sick_rect, day_rect, labels).facet(
        column=alt.Column(
            "month:T",
            title=None,
            header=alt.Header(format="%b %Y", labelAngle=0, labelOrient="top"),
        ),
        columns=3,
    )
    spacer_left, chart_col, spacer_right = st.columns([1, 8, 1])
    with chart_col:
        st.altair_chart(layered, use_container_width=True)

st.subheader("Summary")
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
