"""Application settings loaded from environment variables (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """Whisper model and local API configuration."""

    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    api_host: str
    api_port: int
    max_upload_bytes: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            whisper_model=os.getenv("WHISPER_MODEL", "large-v3-turbo"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            api_host=os.getenv("API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("API_PORT", "8000")),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))),
        )


settings = Settings.from_env()
