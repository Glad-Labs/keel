"""keel serve: talk to Keel by voice, from a browser.

It listens on 127.0.0.1 only, at http://127.0.0.1:8095/. To reach it from a
phone, put it behind `tailscale serve`: that keeps it private to your tailnet
and gives the page the https a browser needs before it allows the microphone.
Add that name to allowed_hosts in ~/.keel/config.toml.

Three guards, because this server holds your journal:

* The Host header must be one Keel expects, so a web page can't reach it by
  pointing its own domain at 127.0.0.1 (DNS rebinding).
* API calls need an X-Keel header. A cross-site request can't send one
  without a preflight, and this server never approves preflights.
* Anything that arrives through a proxy (`tailscale serve` adds forwarding
  headers) must carry a Tailscale login listed in allowed_users. Tailscale
  sets that header itself, so another device on the tailnet can't fake it.

Use `tailscale serve`, never `tailscale funnel`: funnel puts it on the internet.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import ollama
from .config import Config
from .harness import Session

SESSION_IDLE = 30 * 60
MAX_BODY = 25 * 1024 * 1024
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SESSION_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")
EXTENSIONS = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/wav": "wav", "audio/mpeg": "mp3"}


class Keel:
    """Conversations by id, one lock each, and the speech models."""

    def __init__(self, cfg: Config, speech, *, voice: str, stt_model: str, now=None, llm=None):
        self.cfg = cfg
        self.speech = speech
        self.voice = voice
        self.stt_model = stt_model
        self.now = now
        self.llm = llm
        self._sessions: dict[str, list] = {}
        self._lock = threading.Lock()

    def _session(self, sid: str) -> tuple[Session, threading.Lock]:
        with self._lock:
            cutoff = time.monotonic() - SESSION_IDLE
            for stale in [k for k, v in self._sessions.items() if v[2] < cutoff]:
                self._sessions.pop(stale)[0].close()
            if sid not in self._sessions:
                extra = {k: v for k, v in (("now", self.now), ("llm", self.llm)) if v is not None}
                self._sessions[sid] = [Session(self.cfg, session_id=sid, **extra), threading.Lock(), 0.0]
            entry = self._sessions[sid]
            entry[2] = time.monotonic()
            return entry[0], entry[1]

    def save_audio(self, audio: bytes, content_type: str) -> Path:
        self.cfg.audio_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        ext = EXTENSIONS.get(content_type.split(";")[0].strip().lower(), "audio")
        path = self.cfg.audio_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.{ext}"
        path.write_bytes(audio)
        path.chmod(0o600)
        return path

    def hear(self, audio: bytes) -> tuple[str, float]:
        start = time.monotonic()
        return self.speech.transcribe(audio, self.stt_model), time.monotonic() - start

    def answer(self, sid: str, text: str, audio_path: Path | None = None) -> dict:
        session, lock = self._session(sid)
        with lock:
            start = time.monotonic()
            reply = session.turn(text, audio_path=str(audio_path) if audio_path else None)
            think = time.monotonic() - start
        if audio_path is not None and reply.entry_id is None:
            # Questions and commands aren't entries, so their recording isn't kept.
            audio_path.unlink(missing_ok=True)
        start = time.monotonic()
        wav = self.speech.speak(reply.spoken, self.voice)
        return {
            "you": text,
            "keel": reply.text,
            "mode": reply.mode,
            "audio": base64.b64encode(wav).decode(),
            "audio_type": "audio/wav",
            "timing": {"think": round(think, 1), "speak": round(time.monotonic() - start, 1)},
            # For tuning, not shown on the page: how the reply was made.
            "how": {
                "picked": reply.checks.get("picked", []),
                "rewrites": reply.checks.get("revisions", 0),
                "salvage": reply.checks.get("salvage"),
                "left": [str(p) for p in reply.checks.get("remaining", [])],
            },
        }


PROXY_HEADERS = ("X-Forwarded-For", "X-Forwarded-Host", "Forwarded", "Tailscale-User-Login")


def _bare(host: str) -> str:
    host = host.strip().lower()
    return host[1:].split("]")[0] if host.startswith("[") else host.rsplit(":", 1)[0]


class Handler(BaseHTTPRequestHandler):
    server_version = "keel"
    keel: Keel
    allowed_hosts: set[str]
    allowed_users: set[str]

    def log_message(self, fmt, *args):  # keep request lines out of the terminal
        pass

    def _host_ok(self) -> bool:
        if _bare(self.headers.get("Host") or "") not in self.allowed_hosts:
            return False
        if any(self.headers.get(h) for h in PROXY_HEADERS):
            # Came through tailscale serve: it has to be you.
            forwarded = self.headers.get("X-Forwarded-Host")
            if forwarded and _bare(forwarded) not in self.allowed_hosts:
                return False
            login = (self.headers.get("Tailscale-User-Login") or "").strip().lower()
            return bool(login) and login in self.allowed_users
        return True

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if ctype.startswith("text/html"):
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; media-src data: blob:; img-src data:; base-uri 'none'; form-action 'none'",
            )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: dict) -> None:
        self._send(code, json.dumps(data).encode(), "application/json")

    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {"error": "unexpected host"})
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (files("keel") / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/health":
            return self._json(200, {"ok": True, "model": self.keel.cfg.chat.model, "voice": self.keel.voice})
        self._json(404, {"error": "not found"})

    def do_OPTIONS(self):
        self._json(403, {"error": "no cross-site requests"})

    def do_POST(self):
        if not self._host_ok() or self.headers.get("X-Keel") != "1":
            return self._json(403, {"error": "forbidden"})
        url = urlparse(self.path)
        sid = parse_qs(url.query).get("session", [""])[0]
        if not SESSION_ID.match(sid):
            return self._json(400, {"error": "bad session id"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return self._json(413 if length else 400, {"error": "empty or too large"})
        body = self.rfile.read(length)
        try:
            if url.path == "/api/text":
                text = str(json.loads(body).get("text", "")).strip()
                if not text:
                    return self._json(400, {"error": "Say or type something first."})
                return self._json(200, self.keel.answer(sid, text))
            if url.path == "/api/turn":
                audio_path = self.keel.save_audio(body, self.headers.get("Content-Type", "audio/webm"))
                text, heard = self.keel.hear(body)
                if not text:
                    audio_path.unlink(missing_ok=True)
                    return self._json(200, {"error": "I didn't catch that. Try again?"})
                data = self.keel.answer(sid, text, audio_path)
                data["timing"]["hear"] = round(heard, 1)
                return self._json(200, data)
        except ollama.OllamaError as err:
            return self._json(503, {"error": f"Keel couldn't think just now: {err}"})
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad request"})
        self._json(404, {"error": "not found"})


def make_server(
    keel: Keel, host: str = "127.0.0.1", port: int = 8095, allowed_hosts=(), allowed_users=()
) -> ThreadingHTTPServer:
    handler = type(
        "KeelHandler",
        (Handler,),
        {
            "keel": keel,
            "allowed_hosts": LOCAL_HOSTS | {h.lower() for h in allowed_hosts},
            "allowed_users": {u.lower() for u in allowed_users},
        },
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
