"""Keeps sign-in tokens out of the logs.

Clients that can't set WebSocket handshake headers pass the Clerk JWT as
`?token=` (see app.core.security.get_current_user_ws), and uvicorn logs full
request paths -- `"WebSocket /api/v1/battle/match/7/ws?token=eyJ..." [accepted]`
on its error logger, the same for HTTP on its access logger. Anyone who can
read those logs could replay the token until it expires. This filter rewrites
the value to `[REDACTED]` before any handler sees it.
"""
import logging
import re

_TOKEN_PARAM = re.compile(r"([?&](?:token|access_token)=)[^&\s\"']+", re.IGNORECASE)


def redact(text: str) -> str:
    return _TOKEN_PARAM.sub(r"\1[REDACTED]", text)


class RedactTokensFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
        return True


def install() -> None:
    """Attach to the loggers that print request paths. Idempotent."""
    # uvicorn emits request lines directly on these two loggers (logger filters
    # don't apply to records propagated up from children, so they go here).
    for name in ("uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactTokensFilter) for f in logger.filters):
            logger.addFilter(RedactTokensFilter())
