#!/usr/bin/env python3
"""Gym Stats Insights multipage router."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Gym Stats Insights", page_icon="💪", layout="wide")

pages = [
    st.Page("pages/0_Overview.py", title="Overview"),
    st.Page("pages/2_Per_Exercise.py", title="Per-Exercise View"),
]

nav = st.navigation(pages, position="top")
nav.run()
