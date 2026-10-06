"""Optional search by meaning, on the CPU, inside this process.

Both Ollama servers on this machine hold one model at a time, so asking
either one for embeddings would push out the model it's serving. A small CPU
model avoids that. Without fastembed installed, Keel uses keyword search only.
"""

from __future__ import annotations

import math
import os
from array import array

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

_models: dict[str, object] = {}


def available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


def _model(name: str):
    if name not in _models:
        from fastembed import TextEmbedding

        _models[name] = TextEmbedding(model_name=name)
    return _models[name]


def _unit(values) -> array:
    vec = array("f", (float(v) for v in values))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return array("f", (v / norm for v in vec))


def passages(name: str, texts: list[str]) -> list[array]:
    return [_unit(v) for v in _model(name).passage_embed(texts)]


def query(name: str, text: str) -> array:
    return _unit(next(iter(_model(name).query_embed(text))))


def to_blob(vec: array) -> bytes:
    return vec.tobytes()


def from_blob(blob: bytes) -> array:
    vec = array("f")
    vec.frombytes(blob)
    return vec


def dot(a: array, b: array) -> float:
    return sum(x * y for x, y in zip(a, b))
