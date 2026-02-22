from __future__ import annotations

"""Shared UI helpers for the Streamlit civic complaint app.

This module centralises small visual tweaks so all pages share the same
look-and-feel (background, typography, footer, etc.).
"""

import streamlit as st


def check_citizen_access() -> None:
    """Check if user is logged in as citizen. Redirect to home if not."""
    user = st.session_state.get("user")
    if not user:
        st.warning("Please log in first to access this page.")
        st.stop()
    if user.get("role") != "citizen":
        st.error("❌ This page is only for citizens. You are logged in as a " + user.get("role", "unknown") + ".")
        st.stop()


def check_official_access() -> None:
    """Check if user is logged in as official. Redirect to home if not."""
    user = st.session_state.get("user")
    if not user:
        st.warning("Please log in first to access this page.")
        st.stop()
    if user.get("role") != "official":
        st.error("❌ This page is only for municipal officials.")
        st.stop()


def check_admin_access() -> None:
    """Check if user is logged in as admin. Redirect to home if not."""
    user = st.session_state.get("user")
    if not user:
        st.warning("Please log in first to access this page.")
        st.stop()
    if user.get("role") != "admin":
        st.error("❌ This page is only for system administrators.")
        st.stop()


def init_sidebar_language_selector() -> None:
    """Initialize language selector in sidebar - globally available on all pages.
    
    This function should be called early in every page to ensure the language
    selector is available. It syncs the sidebar selection with session state.
    """
    # Initialize language in session if not present
    if "language" not in st.session_state:
        st.session_state["language"] = "English"
    
    # Add language selector to sidebar
    st.sidebar.title("⚙️ Settings")
    lang = st.sidebar.selectbox(
        "Language / भाषा",
        options=["English", "Hindi", "Hinglish"],
        index=["English", "Hindi", "Hinglish"].index(st.session_state["language"]),
    )
    st.session_state["language"] = lang


def apply_global_styles() -> None:
    """Previously injected global CSS.

    This is now a no-op so the app uses Streamlit's default styling.
    Pages can still call this safely without any visual side effects.
    """

    return None


def render_footer() -> None:
    """Render a simple shared footer for all pages.

    Kept minimal (no extra CSS) per user's request.
    """

    st.markdown("---")
    st.caption("© 2025 Municipal IT Desk")
