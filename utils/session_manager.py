from __future__ import annotations

"""Session and access-control utilities for the Streamlit app.

This module centralises logic for:

* Initialising authentication-related keys in ``st.session_state``.
* Creating and invalidating sessions backed by the ``sessions`` table.
* Enforcing a 30-minute inactivity timeout.
* Simple role-based access control helpers for Streamlit pages.

The Streamlit pages should call :func:`init_session_state` early, use
:func:`login_user` after successful authentication, and
:func:`require_auth` at the top of protected pages.
"""

from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import wraps
import logging

import streamlit as st

from .auth import AuthUser, create_session_token, get_active_session, invalidate_session
from .database import get_connection

SESSION_TIMEOUT_MINUTES: int = 30

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Return current UTC time (separate helper for easier testing)."""

    return datetime.utcnow()


def init_session_state() -> None:
    """Initialise authentication-related keys in ``st.session_state``.

    This is safe to call from any page; missing keys are added with
    sensible defaults.
    """

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    if "user_name" not in st.session_state:
        st.session_state.user_name = None

    if "user_email" not in st.session_state:
        st.session_state.user_email = None

    if "user_role" not in st.session_state:
        st.session_state.user_role = None

    if "session_token" not in st.session_state:
        st.session_state.session_token = None

    if "department_id" not in st.session_state:
        st.session_state.department_id = None

    if "login_timestamp" not in st.session_state:
        st.session_state.login_timestamp = None

    if "must_change_password" not in st.session_state:
        st.session_state.must_change_password = False


def set_user_session(user_dict: Dict[str, Any], project_root: Optional[Path] = None) -> None:
    """Helper to adapt dict user records to :class:`AuthUser` and log in."""

    user = AuthUser(
        id=int(user_dict["id"]),
        name=str(user_dict.get("name", "")),
        email=str(user_dict.get("email", "")),
        role=str(user_dict.get("role", "")),
        department_id=user_dict.get("department_id"),
        location=user_dict.get("location"),
    )
    login_user(user, project_root=project_root)


def clear_session(project_root: Optional[Path] = None) -> None:
    """Alias to logout_user for compatibility with existing pages."""

    logout_user(project_root=project_root)


def login_user(user: AuthUser, project_root: Optional[Path] = None) -> None:
    """Mark a user as logged in and create a DB-backed session.

    Args:
        user: Authenticated :class:`AuthUser` instance.
        project_root: Optional project root path.
    """

    init_session_state()
    token = create_session_token(user.id, duration_minutes=SESSION_TIMEOUT_MINUTES, project_root=project_root)

    now = _now_utc().isoformat()
    st.session_state.update(
        {
            "authenticated": True,
            "user_id": user.id,
            "user_name": user.name,
            "user_email": user.email,
            "user_role": user.role,
            "session_token": token,
            "login_timestamp": now,
            "last_activity": now,
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "role": user.role,
                "department_id": user.department_id,
                "location": user.location,
            },
        }
    )


def logout_user(project_root: Optional[Path] = None) -> None:
    """Log out the current user and invalidate their session token."""

    token = st.session_state.get("session_token")
    if token:
        invalidate_session(str(token), project_root=project_root)

    for key in [
        "authenticated",
        "user_id",
        "user_name",
        "user_email",
        "user_role",
        "session_token",
        "login_timestamp",
        "last_activity",
    ]:
        st.session_state[key] = None if key != "authenticated" else False


def _session_expired(last_activity_iso: Optional[str]) -> bool:
    """Return ``True`` if the last activity time exceeds the timeout."""

    if not last_activity_iso:
        return True
    try:
        last_activity = datetime.fromisoformat(str(last_activity_iso))
    except ValueError:
        return True
    return _now_utc() - last_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES)


def get_current_user(project_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the current logged-in user or ``None`` if not authenticated.

    This also enforces the 30-minute inactivity timeout and validates the
    session token against the ``sessions`` table.
    """

    init_session_state()
    if not st.session_state.get("authenticated"):
        return None

    # Inactivity timeout
    if _session_expired(st.session_state.get("last_activity")):
        logout_user(project_root=project_root)
        st.warning("Your session has expired due to inactivity. Please log in again.")
        return None

    token = st.session_state.get("session_token")
    if not token:
        logout_user(project_root=project_root)
        return None

    sess = get_active_session(str(token), project_root=project_root)
    if sess is None:
        logout_user(project_root=project_root)
        return None

    # Refresh last activity timestamp
    st.session_state["last_activity"] = _now_utc().isoformat()

    return {
        "id": st.session_state.get("user_id"),
        "name": st.session_state.get("user_name"),
        "email": st.session_state.get("user_email"),
        "role": st.session_state.get("user_role"),
    }


def require_auth(allowed_roles: Optional[list] = None) -> Callable:
    """
    Decorator to require authentication for a page.

    Args:
        allowed_roles (Optional[list]): List of roles allowed to access

    Returns:
        Callable: Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            init_session_state()

            # Check authentication
            if not is_authenticated():
                st.error("🔒 Please login to access this page")
                st.info("Redirecting to login page...")
                st.stop()

            # Check role-based access
            if allowed_roles:
                user_role = get_user_role()
                if user_role not in allowed_roles:
                    st.error(f"⛔ Access denied. {', '.join(allowed_roles).title()} access required.")
                    st.info(f"You are logged in as: {user_role.title()}")
                    st.stop()

            # Check if password change is required
            if st.session_state.get("must_change_password", False):
                st.warning("🔐 You must change your password before continuing")
                st.stop()

            return func(*args, **kwargs)

        return wrapper

    return decorator


def show_user_info() -> None:
    """Display logged-in user information in sidebar."""
    if is_authenticated():
        st.sidebar.success(f"👤 **{st.session_state.user_name}**")
        st.sidebar.caption(f"Role: {st.session_state.user_role.title()}")
        st.sidebar.caption(f"Email: {st.session_state.user_email}")


def check_complaint_ownership(complaint_id: int, user_id: int) -> bool:
    """
    Verify that a user owns a specific complaint.

    Args:
        complaint_id (int): Complaint ID
        user_id (int): User ID

    Returns:
        bool: True if user owns the complaint, False otherwise
    """
    from .database import execute_query

    try:
        complaint = execute_query(
            "SELECT user_id FROM complaints WHERE id = ?",
            (complaint_id,),
            fetch="one",
        )

        if not complaint:
            return False

        return complaint["user_id"] == user_id
    except Exception as e:
        logger.error(f"Ownership check error: {e}")
        return False


def is_authenticated() -> bool:
    """
    Check if user is authenticated.

    Returns:
        bool: True if authenticated, False otherwise
    """
    return st.session_state.get("authenticated", False)


def get_user_role() -> Optional[str]:
    """
    Get current user's role.

    Returns:
        Optional[str]: User role or None
    """
    return st.session_state.get("user_role")


__all__ = [
    "SESSION_TIMEOUT_MINUTES",
    "init_session_state",
    "login_user",
    "logout_user",
    "set_user_session",
    "clear_session",
    "get_current_user",
    "require_auth",
    "check_complaint_ownership",
    "is_authenticated",
    "get_user_role",
    "show_user_info",
]
