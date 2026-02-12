"""Whisper transcription server using faster-whisper with GPU acceleration."""

import os
import tempfile
from pathlib import Path

# Add all NVIDIA library paths to PATH before importing CUDA-dependent packages
nvidia_base = Path(r"C:\Python314\Lib\site-packages\nvidia")
if nvidia_base.exists():
    for bin_dir in nvidia_base.glob("*/bin"):
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(str(bin_dir))

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper Transcription Server")

# large-v2 is a good balance of speed/accuracy on a 3070
model = WhisperModel("large-v2", device="cuda", compute_type="float16")


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Accept an audio file and return the transcript."""
    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        content = await audio.read()
        with open(fd, "wb") as tmp:
            tmp.write(content)

        segments, info = model.transcribe(tmp_path, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments)

        return {
            "text": text,
            "language": info.language,
            "duration": round(info.duration, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
