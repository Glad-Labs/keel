"""The smallest Ollama client Keel needs, standard library only.

Keel shares the GPUs with Poindexter, so it follows three rules:

* It doesn't load models. If the model isn't already in memory it says so
  and stops, unless you've set allow_load.
* It asks for the context length a model is already loaded with. Ollama
  reloads a model when a request asks for a different one, and a reload in
  the middle of someone else's job is exactly what to avoid.
* It never sends keep_alive. Each Ollama server's own policy (an hour on the
  5090, forever on the 3090) decides how long a model stays loaded.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from functools import lru_cache

from .config import Endpoint, require_local


class OllamaError(RuntimeError):
    pass


class NotLoaded(OllamaError):
    """The model isn't in memory, and Keel won't load it on its own."""


def _request(url: str, path: str, body: dict | None = None, timeout: float = 600) -> dict:
    require_local(url)
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace")[:300]
        raise OllamaError(f"{path} returned {err.code}: {detail}") from err
    except urllib.error.URLError as err:
        raise OllamaError(f"can't reach {url}: {err.reason}") from err


def loaded(url: str) -> dict[str, dict]:
    """Models this Ollama server has in memory right now, by name."""
    return {m["name"]: m for m in _request(url, "/api/ps", timeout=10).get("models", [])}


def installed(url: str) -> set[str]:
    return {m["name"] for m in _request(url, "/api/tags", timeout=10).get("models", [])}


@lru_cache(maxsize=None)
def capabilities(url: str, model: str) -> frozenset[str]:
    info = _request(url, "/api/show", {"model": model}, timeout=30)
    return frozenset(info.get("capabilities", []))


def context_for(ep: Endpoint) -> int:
    model = loaded(ep.url).get(ep.model)
    if model is None:
        if not ep.allow_load:
            raise NotLoaded(
                f"{ep.model} isn't loaded at {ep.url}, and loading it could push out the model "
                "Poindexter is using there. Pick a model that's already loaded, or set allow_load."
            )
        return ep.num_ctx
    return int(model.get("context_length") or ep.num_ctx)


def chat(
    ep: Endpoint,
    messages: list[dict],
    *,
    schema: dict | None = None,
    temperature: float = 0.6,
    timeout: float = 180,
) -> str:
    body = {
        "model": ep.model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": context_for(ep)},
    }
    if "thinking" in capabilities(ep.url, ep.model):
        body["think"] = False
    if schema is not None:
        body["format"] = schema
    return _request(ep.url, "/api/chat", body, timeout)["message"]["content"]


def chat_json(ep: Endpoint, messages: list[dict], schema: dict, **kw) -> dict:
    text = chat(ep, messages, schema=schema, temperature=0, **kw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as err:
        raise OllamaError(f"{ep.model} didn't return JSON: {text[:200]}") from err
