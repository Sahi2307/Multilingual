from __future__ import annotations

"""Authentication and authorization utilities for the Civic Complaint System.
Handles user registration, login, session management, and access control.
"""

import bcrypt
import secrets
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging
from .database import execute_query, insert_record, update_record

logger = logging.getLogger(__name__)

# Security configuration
SESSION_TIMEOUT_MINUTES = 30
PASSWORD_RESET_TOKEN_HOURS = 1
MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_WINDOW_MINUTES = 15


@dataclass
class AuthUser:
    """Lightweight authenticated user container used by session_manager."""

    id: int
    name: str
    email: str
    role: str
    department_id: Optional[int] = None
    location: Optional[str] = None


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt with salt.

    Args:
        password (str): Plain text password

    Returns:
        str: Hashed password
    """

    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        password (str): Plain text password
        password_hash (str): Hashed password

    Returns:
        bool: True if password matches, False otherwise
    """

    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Validate password strength according to security requirements.
    Requirements: min 8 chars, 1 uppercase, 1 number, 1 special char

    Args:
        password (str): Password to validate

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """

    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"

    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number"

    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"

    return True, ""


def validate_email(email: str) -> bool:
    """
    Validate email format.

    Args:
        email (str): Email address

    Returns:
        bool: True if valid, False otherwise
    """

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """
    Validate Indian phone number format.

    Args:
        phone (str): Phone number

    Returns:
        bool: True if valid, False otherwise
    """

    # Indian phone: +91 followed by 10 digits
    pattern = r"^\+91[6-9]\d{9}$"
    return bool(re.match(pattern, phone))


def register_user(
    name: str,
    email: str,
    phone: str,
    password: str,
    role: str = "citizen",
    location: str = "",
    department_id: Optional[int] = None,
    allow_admin_creation: bool = False,
) -> Tuple[bool, str, Optional[int]]:
    """
    Register a new user in the system.

    Args:
        name (str): Full name
        email (str): Email address
        phone (str): Phone number
        password (str): Plain text password
        role (str): User role (citizen, official, admin)
        location (str): User location
        department_id (Optional[int]): Department ID for officials
        allow_admin_creation (bool): Gate to allow creating admin accounts
            (must be True when invoked from an admin-approved flow)

    Returns:
        Tuple[bool, str, Optional[int]]: (success, message, user_id)
    """

    try:
        # Validate inputs
        if not all([name, email, password]):
            return False, "Name, email, and password are required", None

        if not validate_email(email):
            return False, "Invalid email format", None

        if phone and not validate_phone(phone):
            return False, "Invalid phone number format. Use +91XXXXXXXXXX", None

        # Validate password strength
        is_valid, error_msg = validate_password_strength(password)
        if not is_valid:
            return False, error_msg, None

        # Validate role
        allowed_roles = {"citizen", "official", "admin"}
        if role not in allowed_roles:
            return False, f"Invalid role. Allowed roles: {', '.join(allowed_roles)}", None

        # Enforce admin creation policy
        if role == "admin" and not allow_admin_creation:
            return False, "Admin accounts can only be created by an existing admin.", None

        # Check if email already exists
        existing_user = execute_query(
            "SELECT id FROM users WHERE email = ?",
            (email,),
            fetch="one",
        )
        if existing_user:
            return False, "Email already registered", None

        # Hash password
        password_hash = hash_password(password)

        # Determine approval status (citizens auto-approved, officials need approval)
        is_approved = role == "citizen" or role == "admin"

        # Insert user
        user_data = {
            "name": name,
            "email": email,
            "phone": phone or None,
            "password_hash": password_hash,
            "role": role,
            "location": location or None,
            "department_id": department_id,
            "is_active": True,
            "is_approved": is_approved,
            "must_change_password": role == "admin",
            "failed_login_attempts": 0,
            "captcha_required": False,
        }

        user_id = insert_record("users", user_data)

        if user_id:
            approval_msg = "" if is_approved else " Your account is pending admin approval."
            return True, f"Registration successful!{approval_msg}", user_id
        else:
            return False, "Registration failed. Please try again.", None

    except Exception as e:
        logger.error(f"Registration error: {e}")
        return False, f"Registration error: {str(e)}", None


def _record_failed_login(user_id: int) -> None:
    """Increment failed attempts, flag captcha after threshold."""
    try:
        user = execute_query("SELECT failed_login_attempts FROM users WHERE id = ?", (user_id,), fetch="one")
        attempts = int(user.get("failed_login_attempts", 0)) + 1 if user else 1
        update_record(
            "users",
            user_id,
            {
                "failed_login_attempts": attempts,
                "last_failed_login": datetime.now(),
                "captcha_required": attempts >= MAX_LOGIN_ATTEMPTS,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to record login attempt: %s", exc)


def _reset_login_attempts(user_id: int) -> None:
    """Reset failed attempts on successful login or cooldown."""
    try:
        update_record(
            "users",
            user_id,
            {"failed_login_attempts": 0, "captcha_required": False, "last_failed_login": None},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to reset login attempts: %s", exc)


def _requires_captcha(user: Dict, captcha_passed: bool) -> Tuple[bool, str]:
    """Check whether captcha is required based on failed attempts window."""
    attempts = int(user.get("failed_login_attempts", 0) or 0)
    last_failed = user.get("last_failed_login")
    captcha_flag = bool(user.get("captcha_required"))

    if last_failed:
        try:
            last_failed_dt = last_failed if isinstance(last_failed, datetime) else datetime.fromisoformat(str(last_failed))
        except Exception:
            last_failed_dt = datetime.now()
    else:
        last_failed_dt = None

    within_window = False
    if last_failed_dt:
        within_window = datetime.now() - last_failed_dt < timedelta(minutes=LOCKOUT_WINDOW_MINUTES)

    if attempts >= MAX_LOGIN_ATTEMPTS and within_window or captcha_flag:
        if not captcha_passed:
            return True, "Too many failed attempts. Complete CAPTCHA to continue."
    return False, ""


def login_user(
    email: str,
    password: str,
    role: str,
    captcha_passed: bool = False,
) -> Tuple[bool, str, Optional[Dict]]:
    """
    Authenticate user and create session.

    Args:
        email (str): User email
        password (str): Plain text password
        role (str): Expected role
        captcha_passed (bool): Whether CAPTCHA was solved after throttling

    Returns:
        Tuple[bool, str, Optional[Dict]]: (success, message, user_data)
    """

    try:
        # Fetch user
        user = execute_query(
            "SELECT * FROM users WHERE email = ?",
            (email,),
            fetch="one",
        )

        if not user:
            return False, "Invalid email or password", None

        # Rate limiting / CAPTCHA check
        captcha_needed, captcha_msg = _requires_captcha(user, captcha_passed)
        if captcha_needed:
            _record_failed_login(user["id"])
            return False, captcha_msg, None

        # Verify password
        if not verify_password(password, user["password_hash"]):
            _record_failed_login(user["id"])
            return False, "Invalid email or password", None

        # Check if user is active
        if not user["is_active"]:
            return False, "Account is deactivated. Contact admin.", None

        # Check if user is approved (for officials)
        if user["role"] == "official" and not user["is_approved"]:
            return False, "Account is pending admin approval", None

        # Check role match
        if user["role"] != role:
            return False, f"Invalid role. Please login as {user['role']}", None

        # Update last login
        update_record("users", user["id"], {"last_login": datetime.now()})
        _reset_login_attempts(user["id"])

        # Return user data (excluding password_hash)
        user_data = {k: v for k, v in user.items() if k != "password_hash"}

        return True, "Login successful", user_data

    except Exception as e:
        logger.error(f"Login error: {e}")
        return False, f"Login error: {str(e)}", None


def create_session(user_id: int, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Optional[str]:
    """
    Create a new session for authenticated user.

    Args:
        user_id (int): User ID

    Returns:
        Optional[str]: Session token, or None on failure
    """

    try:
        # Generate secure random token
        session_token = secrets.token_urlsafe(32)

        # Calculate expiry time
        expires_at = datetime.now() + timedelta(minutes=SESSION_TIMEOUT_MINUTES)

        # Insert session
        session_data = {
            "user_id": user_id,
            "session_token": session_token,
            "expires_at": expires_at,
            "is_active": True,
            "ip_address": ip_address,
            "user_agent": user_agent,
        }

        insert_record("sessions", session_data)

        return session_token

    except Exception as e:
        logger.error(f"Session creation error: {e}")
        return None


def create_session_token(
    user_id: int,
    duration_minutes: int = SESSION_TIMEOUT_MINUTES,
    project_root: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Optional[str]:
    """Create a session token (API-compatible with session_manager)."""

    try:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=duration_minutes)

        session_data = {
            "user_id": user_id,
            "session_token": token,
            "expires_at": expires_at,
            "is_active": True,
            "ip_address": ip_address,
            "user_agent": user_agent,
        }

        insert_record("sessions", session_data)
        return token
    except Exception as exc:  # noqa: BLE001
        logger.error("Session creation error: %s", exc)
        return None


def get_active_session(session_token: str, project_root: Optional[str] = None) -> Optional[Dict]:
    """Fetch active session + user info for a given token."""

    try:
        return execute_query(
            """
            SELECT s.*, u.*
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_token = ? AND s.is_active = 1 AND s.expires_at > ?
            """,
            (session_token, datetime.now()),
            fetch="one",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Session lookup error: %s", exc)
        return None


def invalidate_session(session_token: str, project_root: Optional[str] = None) -> bool:
    """Mark a session token inactive."""

    try:
        execute_query("UPDATE sessions SET is_active = 0 WHERE session_token = ?", (session_token,))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Session invalidation error: %s", exc)
        return False


def validate_session(session_token: str) -> Optional[Dict]:
    """
    Validate session token and return user data.

    Args:
        session_token (str): Session token

    Returns:
        Optional[Dict]: User data if session is valid, None otherwise
    """

    try:
        # Fetch session
        session = execute_query(
            """
            SELECT s.*, u.* FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_token = ? AND s.is_active = TRUE AND s.expires_at > ?
            """,
            (session_token, datetime.now()),
            fetch="one",
        )

        if not session:
            return None

        # Check if user is active
        if not session["is_active"]:
            return None

        # Return user data
        return {k: v for k, v in session.items() if k != "password_hash"}

    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return None


def logout_user(session_token: str) -> bool:
    """
    Invalidate user session.

    Args:
        session_token (str): Session token

    Returns:
        bool: True if successful, False otherwise
    """

    try:
        query = "UPDATE sessions SET is_active = FALSE WHERE session_token = ?"
        execute_query(query, (session_token,))
        return True
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return False


def change_password(
    user_id: int, old_password: str, new_password: str
) -> Tuple[bool, str]:
    """
    Change user password.

    Args:
        user_id (int): User ID
        old_password (str): Current password
        new_password (str): New password

    Returns:
        Tuple[bool, str]: (success, message)
    """

    try:
        # Fetch user
        user = execute_query(
            "SELECT password_hash FROM users WHERE id = ?",
            (user_id,),
            fetch="one",
        )

        if not user:
            return False, "User not found"

        # Verify old password
        if not verify_password(old_password, user["password_hash"]):
            return False, "Current password is incorrect"

        # Validate new password
        is_valid, error_msg = validate_password_strength(new_password)
        if not is_valid:
            return False, error_msg

        # Hash new password
        new_password_hash = hash_password(new_password)

        # Update password
        update_record("users", user_id, {"password_hash": new_password_hash, "must_change_password": False})

        return True, "Password changed successfully"

    except Exception as e:
        logger.error(f"Password change error: {e}")
        return False, f"Error: {str(e)}"


__all__ = [
    "AuthUser",
    "hash_password",
    "verify_password",
    "validate_password_strength",
    "validate_email",
    "validate_phone",
    "register_user",
    "login_user",
    "create_session",
    "create_session_token",
    "get_active_session",
    "invalidate_session",
    "validate_session",
    "logout_user",
    "change_password",
]
