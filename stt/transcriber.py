"""
stt/transcriber.py
Speech-to-Text module supporting:
  - faster-whisper (local, default)
  - Groq Whisper API
  - OpenAI Whisper API
"""

import os
from pathlib import Path


def transcribe(audio_path: str, config: dict) -> str:
    """
    Transcribe audio file to text.

    Args:
        audio_path: Path to .wav / .mp3 / .ogg file
        config: dict with keys:
            backend: "local" | "groq" | "openai"
            model:   whisper model size (for local) – tiny/base/small/medium/large
            api_key: API key (for groq/openai)

    Returns:
        Transcribed text string
    """
    backend = config.get("backend", "local")

    if backend == "local":
        return _transcribe_local(audio_path, config.get("model", "base"))
    elif backend == "groq":
        return _transcribe_groq(audio_path, config["api_key"])
    elif backend == "openai":
        return _transcribe_openai(audio_path, config["api_key"])
    else:
        raise ValueError(f"Unknown STT backend: {backend}")


# ── Local (faster-whisper) ──────────────────────────────────────────────────────
def _transcribe_local(audio_path: str, model_size: str = "base") -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("Run: pip install faster-whisper")

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


# ── Groq API ───────────────────────────────────────────────────────────────────
def _transcribe_groq(audio_path: str, api_key: str) -> str:
    """
    Uses Groq's hosted Whisper endpoint.
    Groq is very fast (~10x realtime) and free-tier friendly.
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")

    client = Groq(api_key=api_key)
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(Path(audio_path).name, f.read()),
            model="whisper-large-v3",
            response_format="text",
        )
    return transcription.strip()


# ── OpenAI Whisper API ─────────────────────────────────────────────────────────
def _transcribe_openai(audio_path: str, api_key: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
    return transcript.text.strip()
