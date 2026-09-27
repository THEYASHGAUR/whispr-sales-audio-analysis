"""
Transcribe a call without locking the file to one language or one opening speaker.

Each speech region is transcribed on its own, so Hindi, English, and Punjabi in the
same file keep their own language. Two voices are separated by channel (stereo) or
by voice similarity (mono). Either voice may speak first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from sklearn.cluster import KMeans

Role = Literal["agent", "customer"]

_encoder = None
_SAMPLE_RATE = 16_000
_BEAM_SIZE = 5
_TURN_TOP_DB = 32
_TURN_SILENCE_GAP = 0.55
_TURN_MIN_SECONDS = 0.25
_TURN_MAX_SECONDS = 20.0


@dataclass(frozen=True)
class DialogueTurn:
    role: Role
    text: str
    start: float
    end: float
    language: str


def _get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder

        _encoder = VoiceEncoder()
    return _encoder


def wav_channel_count(path: Path) -> int:
    with sf.SoundFile(str(path)) as audio:
        return audio.channels


def _load_channel(path: Path, channel: int | None = None) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=_SAMPLE_RATE, mono=False)
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if channel is None:
        return librosa.to_mono(audio).astype(np.float32)
    index = min(channel, audio.shape[0] - 1)
    return audio[index].astype(np.float32)


def _speech_regions(wav: np.ndarray) -> list[tuple[float, float]]:
    if wav.size == 0:
        return []

    intervals = librosa.effects.split(
        wav,
        top_db=_TURN_TOP_DB,
        frame_length=2048,
        hop_length=512,
    )
    regions: list[tuple[float, float]] = []
    for start_sample, end_sample in intervals:
        start = start_sample / _SAMPLE_RATE
        end = end_sample / _SAMPLE_RATE
        if end - start < _TURN_MIN_SECONDS:
            continue
        if regions and start - regions[-1][1] <= _TURN_SILENCE_GAP:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))

    chopped: list[tuple[float, float]] = []
    for start, end in regions:
        cursor = start
        while end - cursor > _TURN_MAX_SECONDS:
            chopped.append((cursor, cursor + _TURN_MAX_SECONDS))
            cursor += _TURN_MAX_SECONDS
        if end - cursor >= _TURN_MIN_SECONDS:
            chopped.append((cursor, end))
    return chopped


def _decode_turn(model: WhisperModel, clip: np.ndarray) -> tuple[str, str]:
    """Transcribe one region. Language is detected for this region only."""
    if clip.size < int(_TURN_MIN_SECONDS * _SAMPLE_RATE):
        return "", "unknown"

    segments, info = model.transcribe(
        clip,
        task="transcribe",
        beam_size=_BEAM_SIZE,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        without_timestamps=True,
        compression_ratio_threshold=2.2,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    parts = [segment.text.strip() for segment in segments if segment.text.strip()]
    text = _collapse_repeats(" ".join(parts)).strip()
    return text, info.language or "unknown"


def _collapse_repeats(text: str) -> str:
    words = text.split()
    if len(words) < 8:
        return text
    collapsed: list[str] = []
    index = 0
    while index < len(words):
        removed = False
        for size in range(3, min(12, (len(words) - index) // 2) + 1):
            phrase = words[index : index + size]
            if words[index + size : index + 2 * size] != phrase:
                continue
            collapsed.extend(phrase)
            index += 2 * size
            while words[index : index + size] == phrase:
                index += size
            removed = True
            break
        if not removed:
            collapsed.append(words[index])
            index += 1
    return " ".join(collapsed)


def _unique_languages(codes: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for code in codes:
        if code and code != "unknown" and code not in seen:
            seen.append(code)
    return tuple(seen)


def _merge_turns(raw: list[tuple[Role, str, float, float, str]]) -> list[DialogueTurn]:
    if not raw:
        return []

    merged: list[DialogueTurn] = []
    role, text, start, end, language = raw[0]
    buffer = [text]
    languages = [language]

    for next_role, next_text, next_start, next_end, next_language in raw[1:]:
        if next_role == role:
            buffer.append(next_text)
            languages.append(next_language)
            end = next_end
            continue
        merged.append(
            DialogueTurn(
                role=role,
                text=" ".join(buffer).strip(),
                start=start,
                end=end,
                language=", ".join(_unique_languages(languages)),
            )
        )
        role, text, start, end, language = next_role, next_text, next_start, next_end, next_language
        buffer = [next_text]
        languages = [next_language]

    merged.append(
        DialogueTurn(
            role=role,
            text=" ".join(buffer).strip(),
            start=start,
            end=end,
            language=", ".join(_unique_languages(languages)) or "unknown",
        )
    )
    return [turn for turn in merged if turn.text]


def _roles_for_turns(clips: list[np.ndarray]) -> list[Role]:
    """Same voice keeps the same label. The opening turn is not forced to Agent."""
    if len(clips) <= 1:
        return ["agent"]

    encoder = _get_encoder()
    embeddings = []
    usable = []
    for index, clip in enumerate(clips):
        if clip.size < int(0.4 * _SAMPLE_RATE):
            continue
        embeddings.append(encoder.embed_utterance(clip))
        usable.append(index)

    if len(embeddings) < 2:
        return ["agent"] * len(clips)

    predicted = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(np.vstack(embeddings))
    labels: list[int | None] = [None] * len(clips)
    for index, cluster in zip(usable, predicted):
        labels[index] = int(cluster)

    for index, label in enumerate(labels):
        if label is not None:
            continue
        for neighbor in range(index - 1, -1, -1):
            if labels[neighbor] is not None:
                labels[index] = labels[neighbor]
                break
        if labels[index] is None:
            labels[index] = 0

    return ["agent" if label == 0 else "customer" for label in labels]


def _turns_from_waveform(model: WhisperModel, wav: np.ndarray):
    rows: list[tuple[str, float, float, str]] = []
    clips: list[np.ndarray] = []
    for start, end in _speech_regions(wav):
        clip = wav[int(start * _SAMPLE_RATE) : int(end * _SAMPLE_RATE)]
        text, language = _decode_turn(model, clip)
        if not text:
            continue
        rows.append((text, start, end, language))
        clips.append(clip)
    return rows, clips


def _transcribe_mono(model: WhisperModel, wav_path: Path) -> list[DialogueTurn]:
    wav = _load_channel(wav_path)
    rows, clips = _turns_from_waveform(model, wav)
    if not rows:
        return []
    roles = _roles_for_turns(clips)
    tagged = [
        (role, text, start, end, language)
        for (text, start, end, language), role in zip(rows, roles)
    ]
    return _merge_turns(tagged)


def _transcribe_stereo(model: WhisperModel, wav_path: Path) -> list[DialogueTurn]:
    tagged: list[tuple[Role, str, float, float, str]] = []
    # Channels are separate speakers. Who talks first is taken from timestamps.
    for channel, role in ((0, "agent"), (1, "customer")):
        wav = _load_channel(wav_path, channel=channel)
        rows, _clips = _turns_from_waveform(model, wav)
        tagged.extend((role, text, start, end, language) for text, start, end, language in rows)
    tagged.sort(key=lambda row: row[2])
    return _merge_turns(tagged)


def transcribe_with_roles(model: WhisperModel, wav_path: Path) -> tuple[list[DialogueTurn], tuple[str, ...]]:
    if wav_channel_count(wav_path) >= 2:
        turns = _transcribe_stereo(model, wav_path)
    else:
        turns = _transcribe_mono(model, wav_path)
    languages = _unique_languages(
        [code for turn in turns for code in turn.language.split(", ")]
    )
    return turns, languages


def format_dialogue(turns: list[DialogueTurn]) -> str:
    lines: list[str] = []
    for turn in turns:
        label = "Agent" if turn.role == "agent" else "Customer"
        lines.append(f"{label}: {turn.text}")
    return "\n\n".join(lines)
