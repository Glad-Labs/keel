"""Where Keel keeps things, and which local models it talks to.

Settings come from ~/.keel/config.toml if it exists, then from KEEL_*
environment variables (KEEL_CHAT_MODEL, KEEL_CHECK_URL, ...).
"""

from __future__ import annotations

import ipaddress
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class LocalOnlyError(RuntimeError):
    """Raised instead of sending anything off this machine."""


def require_local(url: str) -> str:
    """Return the URL unchanged if it points at this machine, else refuse."""
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return url
    try:
        if ipaddress.ip_address(host).is_loopback:
            return url
    except ValueError:
        pass
    raise LocalOnlyError(
        f"{url} is not on this machine. Keel only talks to models on localhost."
    )


@dataclass(frozen=True)
class Endpoint:
    url: str
    model: str
    # Used only when the model isn't already loaded. Poindexter loads both
    # GPUs' models at 16384, and asking for anything else forces a reload.
    num_ctx: int = 16384
    # Off by default: Keel only uses models that are already in memory. Both
    # Ollama servers here hold one model at a time, so loading one evicts
    # whatever Poindexter is using, and the 5090 is often busy with other GPU
    # work. On 2026-10-06 a Gemma load found 1.3 GB free, spilled onto the
    # CPU, and stalled that server for ten minutes.
    allow_load: bool = False


@dataclass(frozen=True)
class Config:
    home: Path
    chat: Endpoint
    check: Endpoint
    embed_model: str | None

    @property
    def db_path(self) -> Path:
        return self.home / "journal.db"

    @property
    def profile_dir(self) -> Path:
        return self.home / "profile"

    @property
    def proposal_path(self) -> Path:
        return self.home / "proposed.md"

    @property
    def review_marker(self) -> Path:
        return self.home / "last_review"

    @property
    def audio_dir(self) -> Path:
        return self.home / "audio"


DEFAULTS = {
    # The 3090's model is loaded permanently, so using it never loads anything.
    "chat_url": "http://127.0.0.1:11435",
    "chat_model": "qwen3-vl:30b-a3b-instruct",
    "check_url": "http://127.0.0.1:11435",
    "check_model": "qwen3-vl:30b-a3b-instruct",
    "num_ctx": 16384,
    "embed_model": "BAAI/bge-small-en-v1.5",
    "allow_load": False,
}


def load(home: Path | str | None = None, **overrides) -> Config:
    home = Path(home or os.environ.get("KEEL_HOME", "~/.keel")).expanduser()
    values = dict(DEFAULTS)
    config_file = home / "config.toml"
    if config_file.exists():
        values.update(tomllib.loads(config_file.read_text()))
    for key in DEFAULTS:
        env = os.environ.get("KEEL_" + key.upper())
        if env is not None:
            values[key] = env
    values.update({k: v for k, v in overrides.items() if v is not None})

    embed_model = values["embed_model"]
    if not embed_model or str(embed_model).lower() in ("none", "off"):
        embed_model = None
    num_ctx = int(values["num_ctx"])
    allow_load = str(values["allow_load"]).lower() in ("1", "true", "yes")
    return Config(
        home=home,
        chat=Endpoint(require_local(values["chat_url"]), values["chat_model"], num_ctx, allow_load),
        check=Endpoint(require_local(values["check_url"]), values["check_model"], num_ctx, allow_load),
        embed_model=embed_model,
    )
