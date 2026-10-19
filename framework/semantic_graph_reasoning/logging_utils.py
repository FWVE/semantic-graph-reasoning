"""Run artifact and logging helpers that redact credentials by construction."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, SecretStr

_API_KEY_PATTERN = re.compile(r"(?:sk|rk|key)-[A-Za-z0-9_-]{8,}")


def redact_text(value: str) -> str:
    """Remove common credential-shaped tokens before logging persisted text."""
    return _API_KEY_PATTERN.sub("[REDACTED]", value)


def safe_jsonable(value: Any) -> Any:
    """Convert objects to log-safe JSON-compatible values."""
    if isinstance(value, SecretStr):
        return "[REDACTED]"
    if isinstance(value, BaseModel):
        return safe_jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): safe_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_jsonable(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def stable_hash(value: Any) -> str:
    """Hash a canonical, redacted representation for cache and prompt identities."""
    encoded = json.dumps(safe_jsonable(value), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_run_dir(runs_root: Path, run_id: str | None = None) -> Path:
    """Create a unique run directory without touching credentials."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    actual_id = run_id or f"{stamp}-{uuid4().hex[:8]}"
    path = runs_root / actual_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def append_jsonl(path: Path, record: Any) -> None:
    """Append one redacted JSON record to an artifact file."""
    payload = json.dumps(safe_jsonable(record), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
