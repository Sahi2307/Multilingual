from __future__ import annotations

"""Notification utilities.

This module provides simple stubs for:

* Email notifications.
* SMS notifications.
* In-app notifications stored in the database.

In a production deployment you would integrate with real providers
(SMTP, SMS gateway, push notifications, etc.).
"""

import logging
from pathlib import Path
from typing import Optional

from .database import insert_notification

logger = logging.getLogger(__name__)


def send_email_notification(email: str, subject: str, body: str) -> None:
    """Stub for sending an email notification.

    Currently this only logs the notification; plug in an SMTP client or
    transactional email service here.
    """
    logger.info("Email to %s | %s | %s", email, subject, body)


def send_sms_notification(phone: str, message: str) -> None:
    """Stub for sending an SMS notification."""
    logger.info("SMS to %s | %s", phone, message)


def create_in_app_notification(
    user_id: int,
    message: str,
    project_root: Optional[Path] = None,
) -> None:
    """Create an in-app notification stored in the database."""
    insert_notification(user_id=user_id, message=message, project_root=project_root)


__all__ = [
    "send_email_notification",
    "send_sms_notification",
    "create_in_app_notification",
]
