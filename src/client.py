"""Gemini Multimodal Live API WebSocket client and Mock simulation client."""

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Union
import websockets

from src.audio.utils import (
    calculate_rms,
    generate_sine_wave,
    resample_pcm16_linear,
)
from src.config import AssistantConfig

logger = logging.getLogger(__name__)

GEMINI_WS_URL_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)


# Event models emitted by clients
@dataclass
class AudioReceived:
    pcm_data: bytes
    sample_rate: int = 24000


@dataclass
class Interrupted:
    pass


@dataclass
class TurnComplete:
    pass


@dataclass
class TranscriptReceived:
    text: str
    is_user: bool = False


@dataclass
class VoiceActivity:
    event_type: str  # "START" or "END"


LiveEvent = Union[AudioReceived, Interrupted, TurnComplete, TranscriptReceived, VoiceActivity]


class GeminiLiveProtocolError(Exception):
    """Raised when there is a protocol or connection error with Gemini Live API."""


class GeminiLiveClient:
    """Production WebSocket client for Gemini Multimodal Live API (Bidirectional streaming)."""

    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._is_connected = False
        self._receive_task: Optional[asyncio.Task] = None
        self._event_queue: asyncio.Queue[LiveEvent] = asyncio.Queue()

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._ws is not None

    def _get_ws_url(self) -> str:
        if not self.config.api_key:
            raise GeminiLiveProtocolError(
                "Gemini API Key is missing. Provide it via --api-key or set GEMINI_API_KEY environment variable. "
                "Or run with --mock for offline simulation."
            )
        return f"{GEMINI_WS_URL_BASE}?key={self.config.api_key}"

    async def connect(self) -> None:
        """Establish WebSocket connection and complete handshake with Gemini Live API."""
        url = self._get_ws_url()
        logger.info("Connecting to Gemini Live API...")

        try:
            # Connect over secure WebSocket
            self._ws = await websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                max_size=10 * 1024 * 1024,
            )
            self._is_connected = True

            # Send initial BidiGenerateContentSetup handshake
            model_name = self.config.model
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"

            setup_message = {
                "setup": {
                    "model": model_name,
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": self.config.voice_name
                                }
                            }
                        },
                    },
                    "systemInstruction": {
                        "parts": [{"text": self.config.system_instruction}]
                    },
                }
            }

            await self._ws.send(json.dumps(setup_message))
            logger.info("Setup handshake sent, awaiting setupComplete...")

            # Wait for setup confirmation or initial server message
            first_msg_raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            first_msg = json.loads(first_msg_raw)
            if "setupComplete" in first_msg:
                logger.info("Gemini Live session successfully established.")
            else:
                logger.debug("First server message: %s", first_msg)

            # Start background reader task
            self._receive_task = asyncio.create_task(self._listen_loop())

        except websockets.InvalidStatusCode as exc:
            self._is_connected = False
            if exc.status_code == 400 or exc.status_code == 403:
                raise GeminiLiveProtocolError(
                    f"Authentication / API Key error ({exc.status_code}): {exc}.\n"
                    "Please check that your GEMINI_API_KEY is valid and has access to Gemini Live API."
                ) from exc
            raise GeminiLiveProtocolError(f"WebSocket handshake failed: {exc}") from exc
        except Exception as exc:
            self._is_connected = False
            raise GeminiLiveProtocolError(f"Failed to connect to Gemini Live API: {exc}") from exc

    async def send_audio(self, pcm_data: bytes) -> None:
        """Send raw PCM16 16kHz audio chunk to Gemini Live API."""
        if not self._is_connected or self._ws is None:
            raise GeminiLiveProtocolError("Client is not connected to Gemini Live API")

        encoded_audio = base64.b64encode(pcm_data).decode("ascii")
        realtime_input_msg = {
            "realtimeInput": {
                "audio": {
                    "mimeType": f"audio/pcm;rate={self.config.input_sample_rate}",
                    "data": encoded_audio,
                }
            }
        }
        await self._ws.send(json.dumps(realtime_input_msg))

    async def send_audio_stream_end(self) -> None:
        """Notify server that audio stream ended (e.g. mic mute / turn end)."""
        if not self._is_connected or self._ws is None:
            return
        end_msg = {"realtimeInput": {"audioStreamEnd": True}}
        await self._ws.send(json.dumps(end_msg))

    async def _listen_loop(self) -> None:
        """Continuously receive and deserialize incoming server messages."""
        try:
            while self._is_connected and self._ws is not None:
                message_raw = await self._ws.recv()
                data = json.loads(message_raw)
                await self._parse_and_dispatch(data)
        except websockets.ConnectionClosed as exc:
            logger.info("WebSocket connection closed: %s", exc)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Error in Gemini Live receive loop: %s", exc)
        finally:
            self._is_connected = False

    async def _parse_and_dispatch(self, data: dict) -> None:
        """Parse BidiGenerateContentServerMessage fields and enqueue corresponding events."""
        # 1. Voice activity
        if "voiceActivity" in data:
            va_type = data["voiceActivity"].get("type", "")
            if "START" in va_type:
                await self._event_queue.put(VoiceActivity(event_type="START"))
            elif "END" in va_type:
                await self._event_queue.put(VoiceActivity(event_type="END"))

        # 2. Server content
        server_content = data.get("serverContent")
        if server_content:
            # Barge-in signal from server
            if server_content.get("interrupted") is True:
                await self._event_queue.put(Interrupted())

            # Model Turn Audio Parts
            model_turn = server_content.get("modelTurn")
            if model_turn and "parts" in model_turn:
                for part in model_turn["parts"]:
                    inline_data = part.get("inlineData")
                    if inline_data and "data" in inline_data:
                        raw_pcm = base64.b64decode(inline_data["data"])
                        await self._event_queue.put(
                            AudioReceived(pcm_data=raw_pcm, sample_rate=self.config.output_sample_rate)
                        )

            # Transcriptions
            if "inputTranscription" in server_content:
                txt = server_content["inputTranscription"].get("text", "")
                if txt:
                    await self._event_queue.put(TranscriptReceived(text=txt, is_user=True))

            if "outputTranscription" in server_content:
                txt = server_content["outputTranscription"].get("text", "")
                if txt:
                    await self._event_queue.put(TranscriptReceived(text=txt, is_user=False))

            # Turn complete
            if server_content.get("turnComplete") is True:
                await self._event_queue.put(TurnComplete())

    async def receive(self) -> AsyncGenerator[LiveEvent, None]:
        """Yield events as they arrive from the Gemini Live session."""
        while self._is_connected or not self._event_queue.empty():
            try:
                event = await self._event_queue.get()
                yield event
            except asyncio.CancelledError:
                break

    async def close(self) -> None:
        """Close connection and stop listener."""
        self._is_connected = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("Gemini Live client closed")


class MockGeminiLiveClient:
    """Mock Gemini Live client for offline testing, CI, and macOS audio verification."""

    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self._is_connected = False
        self._event_queue: asyncio.Queue[LiveEvent] = asyncio.Queue()
        self._speech_buffer = bytearray()
        self._is_speaking = False
        self._last_speech_time = 0.0
        self._silence_watch_task: Optional[asyncio.Task] = None
        self._mock_player_active = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def connect(self) -> None:
        self._is_connected = True
        logger.info("Mock Gemini Live client initialized (mode=%s)", self.config.mock_type)
        self._silence_watch_task = asyncio.create_task(self._silence_detector())

    async def send_audio(self, pcm_data: bytes) -> None:
        """Receive mic audio and simulate voice activity / echo response."""
        if not self._is_connected:
            return

        rms = calculate_rms(pcm_data)
        threshold_linear = 10 ** (self.config.vad_threshold_db / 20.0)

        # Detect user speech
        if rms > threshold_linear:
            self._last_speech_time = time.monotonic()

            # If mock assistant was currently playing, user speech interrupts it (barge-in!)
            if self._mock_player_active:
                self._mock_player_active = False
                await self._event_queue.put(Interrupted())
                await self._event_queue.put(
                    TranscriptReceived(text="[Barge-in: playback interrupted by user speech]", is_user=False)
                )

            if not self._is_speaking:
                self._is_speaking = True
                await self._event_queue.put(VoiceActivity(event_type="START"))
                await self._event_queue.put(
                    TranscriptReceived(text="[User speaking...]", is_user=True)
                )

            self._speech_buffer.extend(pcm_data)
        elif self._is_speaking:
            # Continue buffering speech trailing silence
            self._speech_buffer.extend(pcm_data)

    async def _silence_detector(self) -> None:
        """Monitor silence to detect end-of-turn and generate mock responses."""
        while self._is_connected:
            await asyncio.sleep(0.05)
            if self._is_speaking:
                elapsed = time.monotonic() - self._last_speech_time
                # 0.5s of silence after speech -> turn complete & trigger mock reply
                if elapsed > 0.5:
                    self._is_speaking = False
                    await self._event_queue.put(VoiceActivity(event_type="END"))
                    await self._generate_mock_response()

    async def _generate_mock_response(self) -> None:
        """Generate simulated audio reply (echo or sine tone) in streaming chunks."""
        buffered_audio = bytes(self._speech_buffer)
        self._speech_buffer.clear()

        if len(buffered_audio) < 1000:
            return

        self._mock_player_active = True

        if self.config.mock_type == "sine":
            # Generate synthesized pleasant chime chords (440Hz + 880Hz)
            await self._event_queue.put(
                TranscriptReceived(text="[Assistant: Simulated 440Hz/880Hz audio response]", is_user=False)
            )
            sine1 = generate_sine_wave(frequency_hz=440.0, duration_sec=0.8, sample_rate=self.config.output_sample_rate)
            sine2 = generate_sine_wave(frequency_hz=880.0, duration_sec=0.8, sample_rate=self.config.output_sample_rate)
            # Combine audio
            arr1 = np.frombuffer(sine1, dtype=np.int16).astype(np.float32)
            arr2 = np.frombuffer(sine2, dtype=np.int16).astype(np.float32)
            combined = np.clip((arr1 + arr2) * 0.5, -32768, 32767).astype(np.int16).tobytes()
            response_audio = combined
        else:
            # Echo mode: Resample user captured speech from 16kHz to 24kHz
            await self._event_queue.put(
                TranscriptReceived(text="[Assistant: Echoing your voice back at 24kHz]", is_user=False)
            )
            response_audio = resample_pcm16_linear(
                buffered_audio,
                src_rate=self.config.input_sample_rate,
                dst_rate=self.config.output_sample_rate,
            )

        # Stream audio chunks with realistic timing
        chunk_size = self.config.block_size * 2  # 1024 samples = 2048 bytes
        for offset in range(0, len(response_audio), chunk_size):
            if not self._mock_player_active or not self._is_connected:
                # Interrupted mid-stream!
                break
            chunk = response_audio[offset : offset + chunk_size]
            await self._event_queue.put(
                AudioReceived(pcm_data=chunk, sample_rate=self.config.output_sample_rate)
            )
            # Sleep corresponding to real playback time: ~42ms for 1024 samples at 24kHz
            chunk_duration = len(chunk) / (self.config.output_sample_rate * 2)
            await asyncio.sleep(min(0.04, chunk_duration))

        if self._mock_player_active:
            self._mock_player_active = False
            await self._event_queue.put(TurnComplete())

    async def receive(self) -> AsyncGenerator[LiveEvent, None]:
        """Yield mock events to listener."""
        while self._is_connected or not self._event_queue.empty():
            try:
                event = await self._event_queue.get()
                yield event
            except asyncio.CancelledError:
                break

    async def close(self) -> None:
        self._is_connected = False
        self._mock_player_active = False
        if self._silence_watch_task and not self._silence_watch_task.done():
            self._silence_watch_task.cancel()
            try:
                await self._silence_watch_task
            except asyncio.CancelledError:
                pass
            self._silence_watch_task = None
        logger.info("Mock Gemini Live client closed")


def create_client(config: AssistantConfig) -> Union[GeminiLiveClient, MockGeminiLiveClient]:
    """Factory creating either real GeminiLiveClient or MockGeminiLiveClient based on config."""
    if config.mock:
        return MockGeminiLiveClient(config)
    return GeminiLiveClient(config)
