"""
Local WAV transcription using faster-whisper (CPU, int8).

Designed for reuse from a CLI or a background worker: load the model once,
then call transcribe_wav() for each file.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from config import settings
from speaker_separation import DialogueTurn, format_dialogue, transcribe_with_roles


@dataclass(frozen=True)
class TranscriptionResult:
    """Structured output for downstream workers (queues, APIs, logs)."""

    languages: tuple[str, ...]
    processing_time_seconds: float
    text: str
    turns: tuple[DialogueTurn, ...]


def load_model() -> WhisperModel:
    """Load Whisper using device/compute_type/model from .env (see config.py)."""
    return WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


def transcribe_wav(model: WhisperModel, wav_path: str | Path) -> TranscriptionResult:
    """
    Transcribe a single .wav file and return language, elapsed time, and full text.

    Args:
        model: A loaded WhisperModel instance (reuse across files in a worker).
        wav_path: Path to a .wav audio file.

    Returns:
        TranscriptionResult with detected language, wall-clock transcribe time, and transcript.
    """
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got: {path.suffix}")

    start = time.perf_counter()
    turns, languages = transcribe_with_roles(model, path)
    dialogue_text = format_dialogue(turns)
    plain_text = " ".join(turn.text for turn in turns).strip()
    elapsed = time.perf_counter() - start

    return TranscriptionResult(
        languages=languages,
        processing_time_seconds=elapsed,
        text=dialogue_text or plain_text,
        turns=tuple(turns),
    )


def format_report(result: TranscriptionResult) -> str:
    """Human-readable report for CLI or logging."""
    return (
        f"Languages: {', '.join(result.languages) or 'unknown'}\n"
        f"Processing time: {result.processing_time_seconds:.2f}s\n"
        f"Transcript:\n{result.text}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a .wav file with faster-whisper (large-v3-turbo, CPU int8).",
    )
    parser.add_argument(
        "wav_path",
        type=Path,
        help="Path to the input .wav audio file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model = load_model()
        result = transcribe_wav(model, args.wav_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
