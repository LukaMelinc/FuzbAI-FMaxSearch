import logging
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def setup_logging(level: Optional[str] = None, *, log_file: Optional[str] = None) -> None:
    """
    Call once near your entrypoint (e.g. in __main__).
    Env vars:
      - LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
      - LOG_FILE=/path/to/file.log  (optional)
    """
    root = logging.getLogger()
    if getattr(root, "_fuzbai_configured", False):
        return

    effective_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    effective_log_file = log_file or os.getenv("LOG_FILE")

    handlers = []

    # Colored console logs if rich is installed; fallback otherwise
    try:
        from rich.logging import RichHandler  # pip install rich
        handlers.append(RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False))
        fmt = "%(message)s"
    except Exception:
        handlers.append(logging.StreamHandler())
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    if effective_log_file:
        fh = logging.FileHandler(effective_log_file)
        fh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handlers.append(fh)

    logging.basicConfig(level=effective_level, format=fmt, handlers=handlers)
    root._fuzbai_configured = True