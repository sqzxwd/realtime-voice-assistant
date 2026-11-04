# Realtime Voice Assistant (`realtime-voice-assistant`)

[![Tests](https://img.shields.io/badge/tests-32%20passed%20(100%25)-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Production-ready Python application for **bidirectional, real-time voice streaming** with Google's **Gemini Multimodal Live API** (`gemini-3.8-live`).

Features low-latency full-duplex audio, jitter-free playback buffering, instant interruption (**barge-in**), macOS microphone permission diagnostics, a rich terminal status UI with VU-meters, and an offline **simulation mode** (`--mock`) requiring no API keys.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │              Audio Pipeline                  │
                      └──────────────────────────────────────────────┘

  [Microphone] 16 kHz Mono PCM16
       │
       ▼
  sounddevice.InputStream (C Audio Thread)
       │  (threadsafe enqueue)
       ▼
  asyncio.Queue (AudioCapture)
       │
       ├──► Local VAD / RMS Calculation (VU Meter & Local Barge-In Trigger)
       ▼
  GeminiLiveClient (WebSocket / Raw BidiGenerateContent Protocol)
       │
       │  wss://generativelanguage.googleapis.com/.../BidiGenerateContent
       │  BidiGenerateContentRealtimeInput (audio/pcm;rate=16000, base64)
       ▼
   [Gemini Live API]
       │
       │  BidiGenerateContentServerMessage
       │  ├── serverContent.modelTurn.parts[].inlineData (24 kHz PCM)
       │  ├── serverContent.interrupted (barge-in event)
       │  └── inputTranscription / outputTranscription
       ▼
  AudioPlayer (Jitter Ring Buffer)
       │
       ├──► Instant Clear on Barge-in (Buffer Flush)
       ├──► Pre-buffering (~60ms) to eliminate network jitter & glitches
       ▼
  sounddevice.OutputStream (C Audio Thread)
       │
       ▼
  [Speakers / Headphones] 24 kHz Mono PCM16
```

### Key Components

- **`src/audio/capture.py` (`AudioCapture`)**:
  Asynchronous raw audio capture from the microphone at 16 kHz, 16-bit mono PCM. Uses `sounddevice.InputStream` in PortAudio's dedicated OS thread and transfers frames via `call_soon_threadsafe` into an `asyncio.Queue`. Automatically drops stale chunks under backpressure to avoid latency ballooning.
- **`src/audio/player.py` (`AudioPlayer`)**:
  Streaming audio player with a jitter ring buffer. Plays 24 kHz (Gemini native rate) or 16 kHz audio without stuttering. Includes pre-buffering to absorb WebSocket delivery jitter and a thread-safe `clear()` method for sub-millisecond barge-in interruption.
- **`src/audio/utils.py`**:
  Vectorized int16/float32 PCM conversions, RMS volume computation, dB conversion, terminal VU-meter rendering, pure sine wave generation with click-free fade envelopes, and linear interpolation resampling.
- **`src/client.py` (`GeminiLiveClient` & `MockGeminiLiveClient`)**:
  WebSocket client communicating with the Gemini Multimodal Live API via raw JSON frames (`BidiGenerateContentSetup`, `realtimeInput`, `serverContent`, `voiceActivity`). Includes `MockGeminiLiveClient` for offline simulation without an API key.
- **`src/cli.py`**:
  Entrypoint built with `typer` and `rich.live.Live`. Displays real-time status (`LISTENING`, `STREAMING`, `SPEAKING`, `BARGE-IN`), dual VU-meters for microphone and speakers in dB, and conversation transcripts.
- **`src/config.py` (`AssistantConfig`)**:
  Type-safe configuration validated with Pydantic v2.

---

## Barge-in (Interruption) Handling

Natural conversations require instant interruption when the user speaks while the assistant is talking. This application implements **dual-path barge-in**:

1. **Local VAD Barge-in (Zero-Latency)**:
   When `player.is_playing` is True and the local microphone signal exceeds `vad_threshold_db` (-36 dB default), `player.clear()` is called immediately. All queued speaker audio is discarded within less than 1 ms, before waiting for a network round-trip.
2. **Server-Side Barge-in**:
   When the Gemini server detects user activity, it emits `serverContent.interrupted = True` and `voiceActivity = {"type": "VOICE_ACTIVITY_START"}`. The client dispatches an `Interrupted` event that flushes the player buffer and halts active generation.

---

## macOS Microphone Permissions

On macOS, terminal applications require explicit permission to access audio input devices via TCC (Transparency, Consent, and Control).

If permissions are denied or unavailable:
1. `AudioCapture` catches the PortAudio host error.
2. A friendly diagnostic panel is displayed:
   ```
   Microphone Access Denied on macOS
   1. Open System Settings -> Privacy & Security -> Microphone
   2. Enable access for Terminal / iTerm2 / VSCode
   3. Restart the terminal and try again.
   ```
3. To reset permissions if needed, run in macOS Terminal:
   ```bash
   tccutil reset Microphone
   ```

---

## Installation

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- macOS / Linux / Windows with audio input & output devices
- PortAudio (bundled automatically with `sounddevice` wheels)

### Setup with `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/username/realtime-voice-assistant.git
cd realtime-voice-assistant

# Create venv and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Setup with standard `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quick Start & Usage

### 1. Offline Simulation Mode (`--mock`)
Verify your macOS audio drivers, microphone, speakers, and barge-in logic without a Gemini API key:

```bash
# Echo mode: records your speech at 16kHz and echoes it back at 24kHz
realtime-voice-assistant --mock --mock-type echo

# Tone mode: generates synthesized 440Hz/880Hz audio chords on voice detection
realtime-voice-assistant --mock --mock-type sine
```

> **Tip**: While the mock assistant is echoing or playing, speak into the microphone to observe instant `⚡ BARGE-IN` cutoff!

### 2. Live Mode with Gemini Live API

Obtain an API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
export GEMINI_API_KEY="your-api-key-here"

# Start assistant with default model (gemini-3.8-live) and voice (Puck)
realtime-voice-assistant

# Or pass custom model and voice:
realtime-voice-assistant --model gemini-3.8-live --voice Aoede
```

### 3. List Available Audio Devices

```bash
realtime-voice-assistant --list-devices
```

Output:
```
  ID   Name                     In Channels   Out Channels   Default  
 ──────────────────────────────────────────────────────────────────── 
   0   iPhone 11 Microphone         1             0            -     
   1   MacBook Air Microphone        1             0          INPUT   
   2   MacBook Air Speakers          0             2         OUTPUT   
```

---

## CLI Options Reference

| Flag | Default | Description |
|---|---|---|
| `-m, --mock` | `False` | Run in offline simulation mode without API key |
| `--mock-type` | `echo` | Mock response mode: `echo` or `sine` |
| `-k, --api-key` | `None` / `$GEMINI_API_KEY` | Gemini API key |
| `--model` | `gemini-3.8-live` | Gemini Multimodal Live API model |
| `--voice` | `Puck` | Prebuilt voice: `Puck`, `Charon`, `Kore`, `Fenrir`, `Aoede` |
| `--vad-threshold`| `-36.0` | Mic volume threshold in dB to trigger voice activity |
| `--barge-in / --no-barge-in` | `True` | Enable/disable instant interruption |
| `-l, --list-devices` | `False` | Print available audio devices and exit |
| `--help` | | Show help message and exit |

---

## Running Unit Tests

The test suite covers:
- Audio player ring buffer, jitter prebuffering, and underflow silence padding.
- Instant barge-in clearing and sample zeroing.
- Vectorized PCM16 conversions, RMS volume measurement, and linear interpolation resampling.
- Gemini Live WebSocket message serialization & deserialization (`BidiGenerateContent`).
- Mock client speech detection, echo streaming, and interruption handling.
- macOS PortAudio error handling and permission diagnostics.
- Typer CLI argument parsing and Rich UI state transitions.

Run the test suite with `pytest`:

```bash
pytest -v
```

Output:
```
tests/test_audio_buffer.py .....                                         [ 15%]
tests/test_capture_and_player_mocked.py ....                             [ 28%]
tests/test_cli.py .....                                                  [ 43%]
tests/test_client_protocol.py .......                                    [ 65%]
tests/test_mock_client.py ...                                            [ 75%]
tests/test_pcm_utils.py ........                                         [100%]

============================== 32 passed in 0.74s ==============================
```

---

## Project Structure

```
realtime-voice-assistant/
├── pyproject.toml              # Build config, dependencies & entrypoints
├── .gitignore                  # Git ignore rules
├── README.md                   # Documentation and architecture guide
├── src/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI entrypoint & Rich terminal UI
│   ├── client.py               # Gemini Live WebSocket & Mock client
│   ├── config.py               # Pydantic configuration schemas
│   └── audio/
│       ├── __init__.py
│       ├── capture.py          # sounddevice.InputStream raw audio capture (16kHz)
│       ├── player.py           # sounddevice.OutputStream jitter buffer (24kHz)
│       └── utils.py            # PCM utils, RMS, VU meter, sine wave generator
└── tests/
    ├── __init__.py
    ├── conftest.py             # Pytest fixtures
    ├── test_audio_buffer.py    # Player buffer, prebuffering & barge-in tests
    ├── test_capture_and_player_mocked.py # macOS permission & capture tests
    ├── test_cli.py             # CLI flags & UI state tests
    ├── test_client_protocol.py # WebSocket protocol & schema tests
    ├── test_mock_client.py     # Offline simulation & barge-in tests
    └── test_pcm_utils.py       # Audio mathematics & VU meter tests
```

---

## License

MIT License.
