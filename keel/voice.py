"""Speech in and out, on the CPU, inside this process.

Whisper turns what you said into text and Kokoro reads Keel's reply aloud.
Both run on the CPU so they never compete with the GPUs (Poindexter's speech
server lives on the busy 5090). The model files are the ones already in the
Hugging Face cache on this machine; nothing is sent anywhere.

Needs the voice extra: uv pip install -e ".[voice]"
"""

from __future__ import annotations

import io
import os
import threading
import wave

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

STT_MODEL = "Systran/faster-whisper-medium"
TTS_REPO = "speaches-ai/Kokoro-82M-v1.0-ONNX"
TTS_VOICE = "af_heart"
CPU_THREADS = 8  # half the cores, leaving the rest for everything else

_lock = threading.Lock()
_stt = {}
_tts = {}


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        import kokoro_onnx  # noqa: F401
    except ImportError:
        return False
    return True


def _snapshot(repo: str) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        # Not cached yet: a one-time download of public model files.
        return snapshot_download(repo)


def _whisper(name: str):
    if name not in _stt:
        from faster_whisper import WhisperModel

        _stt[name] = WhisperModel(_snapshot(name), device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)
    return _stt[name]


def _kokoro():
    if "model" not in _tts:
        from kokoro_onnx import Kokoro

        snap = _snapshot(TTS_REPO)
        _tts["model"] = Kokoro(os.path.join(snap, "model.onnx"), os.path.join(snap, "voices.bin"))
    return _tts["model"]


def voices() -> list[str]:
    return list(_kokoro().get_voices())


def transcribe(audio: bytes, model: str = STT_MODEL) -> str:
    """Audio in any format ffmpeg reads (the browser sends webm/opus) to text."""
    with _lock:
        segments, _info = _whisper(model).transcribe(
            io.BytesIO(audio), language="en", beam_size=1, vad_filter=True
        )
        return " ".join(s.text.strip() for s in segments).strip()


def speak(text: str, voice: str = TTS_VOICE, speed: float = 1.0) -> bytes:
    """Text to a WAV file's bytes, 24 kHz mono."""
    import numpy as np

    with _lock:
        samples, rate = _kokoro().create(text, voice=voice, speed=speed, lang="en-us")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


def warm(model: str = STT_MODEL) -> None:
    """Load both models now, so the first turn isn't the slow one."""
    _whisper(model)
    _kokoro()
