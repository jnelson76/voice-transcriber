"""Voice notes recorder - captures audio, transcribes via Whisper, formats with Ollama, saves to Obsidian."""

import io
import json
import wave
import sys
from datetime import datetime
from pathlib import Path

import requests
import sounddevice as sd
import numpy as np

# --- Configuration ---
WHISPER_URL = "http://192.168.10.163:8090/transcribe"
OLLAMA_URL = "http://192.168.10.163:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"
OBSIDIAN_VAULT = Path(r"C:\Users\SyncthingServiceAcct\mine\mine\Wiser\Voice Notes")
SAMPLE_RATE = 16000  # 16kHz is ideal for Whisper

MEETING_PROMPT = """You are a meeting notes formatter. Given a raw voice transcript, produce clean structured meeting notes in markdown. Include these sections only if relevant content exists:

## Attendees
- (list if mentioned)

## Key Points
- (main topics discussed)

## Action Items
- [ ] (tasks assigned, with owner if mentioned)

## Decisions
- (decisions made)

## Notes
- (anything else noteworthy)

Keep it concise. Do not add information that wasn't in the transcript. If the transcript is short or informal, keep the notes proportionally brief.

Raw transcript:
{transcript}"""


def record_audio() -> np.ndarray:
    """Record audio from microphone until user presses Enter."""
    print("\n  Recording... press Enter to stop.\n")
    frames = []
    recording = True

    def callback(indata, frame_count, time_info, status):
        if recording:
            frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback)
    stream.start()

    input()  # Block until Enter
    recording = False
    stream.stop()
    stream.close()

    if not frames:
        return np.array([], dtype="int16")

    return np.concatenate(frames, axis=0)


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf.read()


def transcribe(wav_bytes: bytes) -> dict:
    """Send audio to Whisper server for transcription."""
    print("  Transcribing...")
    resp = requests.post(
        WHISPER_URL,
        files={"audio": ("recording.wav", wav_bytes, "audio/wav")},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def format_notes(transcript: str) -> str:
    """Send transcript to Ollama for meeting note formatting."""
    print("  Formatting notes with LLM...")
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": MEETING_PROMPT.format(transcript=transcript),
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def save_notes(formatted: str, transcript: str, duration: float) -> Path:
    """Save formatted notes to Obsidian vault."""
    OBSIDIAN_VAULT.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = now.strftime("%Y-%m-%d-%H%M") + "-meeting-notes.md"
    filepath = OBSIDIAN_VAULT / filename

    content = f"""# Meeting Notes - {now.strftime("%Y-%m-%d %H:%M")}

> Duration: {duration}s | Transcribed with Whisper + llama3.1

{formatted}

---

<details>
<summary>Raw Transcript</summary>

{transcript}

</details>
"""
    filepath.write_text(content, encoding="utf-8")
    return filepath


def main():
    print("=" * 50)
    print("  Voice Notes Recorder")
    print("=" * 50)
    print(f"  Whisper: {WHISPER_URL}")
    print(f"  Ollama:  {OLLAMA_MODEL}")
    print(f"  Output:  {OBSIDIAN_VAULT}")

    while True:
        print("\n  Press Enter to start recording (or 'q' to quit):")
        choice = input("  > ").strip().lower()
        if choice == "q":
            print("  Bye!")
            break

        audio = record_audio()
        if audio.size == 0:
            print("  No audio captured, try again.")
            continue

        duration_sec = round(len(audio) / SAMPLE_RATE, 1)
        print(f"  Captured {duration_sec}s of audio.")

        wav_bytes = audio_to_wav_bytes(audio)

        try:
            result = transcribe(wav_bytes)
            transcript = result["text"]
            print(f"\n  Transcript ({result['language']}, {result['duration']}s):")
            print(f"  {transcript[:200]}{'...' if len(transcript) > 200 else ''}\n")

            formatted = format_notes(transcript)
            filepath = save_notes(formatted, transcript, result["duration"])
            print(f"  Saved to: {filepath}")

        except requests.ConnectionError:
            print("  ERROR: Cannot reach server. Is the Whisper server running?")
        except requests.HTTPError as e:
            print(f"  ERROR: Server returned {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
