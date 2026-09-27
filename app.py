"""
FastAPI server: upload a .wav file and return the transcript as JSON.

Run: uvicorn app:app --host 127.0.0.1 --port 8000
Or:  python app.py
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

from config import settings
from transcribe_wav import TranscriptionResult, load_model, transcribe_wav

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup so each request does not reload weights.
    app.state.whisper_model = load_model()
    yield
    app.state.whisper_model = None


app = FastAPI(
    title="Local Whisper Transcription",
    description="Transcribe .wav files with faster-whisper (large-v3-turbo).",
    lifespan=lifespan,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    """Serve the upload UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.whisper_model}


def result_to_json(result: TranscriptionResult) -> dict:
    return {
        "languages": list(result.languages),
        "processing_time_seconds": round(result.processing_time_seconds, 3),
        "text": result.text,
        "turns": [
            {
                "role": turn.role,
                "label": "Agent" if turn.role == "agent" else "Customer",
                "language": turn.language,
                "text": turn.text,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            }
            for turn in result.turns
        ],
    }


@app.post("/api/transcribe")
async def transcribe_endpoint(request: Request, file: UploadFile = File(...)):
    """
    Accept a multipart .wav upload, transcribe it, and return JSON.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    body = await file.read()
    if len(body) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size ({settings.max_upload_bytes} bytes)",
        )
    if not body:
        raise HTTPException(status_code=400, detail="Empty file")

    model: WhisperModel = request.app.state.whisper_model
    suffix = Path(file.filename).suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    try:
        result = transcribe_wav(model, tmp_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {exc}",
        ) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result_to_json(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
