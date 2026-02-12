# Voice Transcriber

Local voice-to-meeting-notes pipeline using Whisper (STT) and Ollama (LLM), with output to an Obsidian vault.

```
Windows mic → Whisper (GPU) → Ollama llama3.1 → Obsidian markdown
```

## Architecture

| Component | Runs on | Purpose |
|-----------|---------|---------|
| `server/whisper_server.py` | Machine with GPU + Ollama | FastAPI server, transcribes audio with faster-whisper on GPU |
| `client/voice_notes.py` | Windows workstation | Records mic, orchestrates pipeline, writes notes to Obsidian |

## Setup

### Server (Ollama machine)

```bash
# Install dependencies
pip install -r server/requirements.txt

# First run will download the Whisper large-v2 model (~3GB)
python server/whisper_server.py
# Listening on http://0.0.0.0:8090
```

Requires CUDA-capable GPU (tested on RTX 3070).

### Client (Windows)

```powershell
# Install dependencies
pip install -r client/requirements.txt

# Run
python client/voice_notes.py
```

Notes are saved to the path configured in `OBSIDIAN_VAULT` (defaults to `~/Documents/Voice Notes`).

## Usage

1. Start the Whisper server on the Ollama machine
2. Run `voice_notes.py` on Windows
3. Press Enter to start recording
4. Press Enter to stop recording
5. Wait for transcription + formatting
6. Notes appear in Obsidian vault

## Configuration

Edit the constants at the top of `client/voice_notes.py`:

All settings can be overridden via environment variables or by editing the constants in `client/voice_notes.py`:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `WHISPER_URL` | `http://localhost:8090/transcribe` | Whisper server endpoint |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.1:latest` | Model for note formatting |
| `OBSIDIAN_VAULT` | `~/Documents/Voice Notes` | Output directory |

Sample rate (16kHz, optimal for Whisper) is set in `voice_notes.py`.

## Running the Server as a Service (optional)

To keep the Whisper server running on the Ollama machine:

```bash
# Using systemd
sudo tee /etc/systemd/system/whisper-server.service << 'EOF'
[Unit]
Description=Whisper Transcription Server
After=network.target

[Service]
User=YOUR_USER
WorkingDirectory=/path/to/voice-transcriber/server
ExecStart=/usr/bin/python3 whisper_server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now whisper-server
```
