"""Telegram bot initialization and token validation helpers."""

from __future__ import annotations

import re
from typing import Callable


TELEGRAM_TOKEN_RE = re.compile(r"^\d{6,}:[A-Za-z0-9_-]{20,}$")


def validate_telegram_token(token: str | None) -> bool:
    if not token:
        return False
    return bool(TELEGRAM_TOKEN_RE.match(str(token).strip()))


def sanitize_telegram_error(exc: Exception) -> str:
    text = str(exc)
    text = re.sub(r"\d{6,}:[A-Za-z0-9_-]{20,}", "<redacted-token>", text)
    return text


def create_telegram_bot(token: str | None, log: Callable[[str], None] | None = None):
    """Return a TeleBot instance or None without leaking sensitive values."""
    logger = log or (lambda _msg: None)
    if not validate_telegram_token(token):
        logger("[Telegram] Bot disabled: TELEGRAM_TOKEN is missing or has an invalid format.")
        return None
    try:
        import telebot

        return telebot.TeleBot(str(token).strip())
    except Exception as exc:
        logger(f"[Telegram] Bot initialization failed: {sanitize_telegram_error(exc)}")
        return None
