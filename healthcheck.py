"""Container healthcheck for the quant system."""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        for path in ("config", "models", "reports", "logs"):
            os.makedirs(path, exist_ok=True)

        from quant import bollinger, data_sources, features, scoring, technical, vcp  # noqa: F401
        from quant.telegram_bot import validate_telegram_token

        token = os.environ.get("TELEGRAM_TOKEN", "")
        if token and not validate_telegram_token(token):
            print("unhealthy: TELEGRAM_TOKEN format is invalid")
            return 1

        data_sources.cache.clear_expired()
        print("healthy")
        return 0
    except Exception as exc:
        print(f"unhealthy: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

