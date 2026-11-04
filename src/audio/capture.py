"""Asynchronous raw audio capture from microphone using sounddevice.InputStream."""

import asyncio
import logging
from typing import AsyncGenerator, Optional
import numpy as np
import sounddevice as sd

from src.audio.utils import calculate_rms, rms_to_db

logger = logging.getLogger(__name__)


class MicrophoneError(Exception):
    """Base exception for audio capture errors."""


class MicrophonePermissionError(MicrophoneError):
    """Raised when macOS microphone permission is denied or access is blocked."""


class AudioCapture:
    """Captures raw 16-bit PCM audio asynchronously from the microphone."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 1024,
        device: Optional[int | str] = None,
        max_queue_size: int = 50,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device = device
        self.max_queue_size = max_queue_size

        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_queue_size)
        self._stream: Optional[sd.InputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._is_running = False
        self._current_rms: float = 0.0

    @property
    def is_running(self) -> bool:
        """Return True if capture stream is active."""
        return self._is_running

    def get_current_rms(self) -> float:
        """Get the latest normalized RMS audio level [0.0, 1.0]."""
        return self._current_rms

    def get_current_db(self, min_db: float = -60.0) -> float:
        """Get the latest audio level in decibels [min_db, 0.0]."""
        return rms_to_db(self._current_rms, min_db=min_db)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """PortAudio C callback executed in a dedicated OS audio thread."""
        if status:
            logger.warning("Audio capture status flag: %s", status)

        # Convert to raw PCM16 bytes
        chunk = indata.tobytes()
        self._current_rms = calculate_rms(chunk)

        if not self._is_running or self._loop is None or self._loop.is_closed():
            return

        def _enqueue() -> None:
            if self._queue.full():
                try:
                    # Drop oldest chunk to prevent unbounded buffer delay/latency
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

        try:
            self._loop.call_soon_threadsafe(_enqueue)
        except RuntimeError:
            pass

    async def start(self) -> None:
        """Start microphone capture stream."""
        if self._is_running:
            return

        self._loop = asyncio.get_running_loop()

        # Drain any leftover queue data
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.block_size,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_running = True
            logger.info(
                "Audio capture started: %d Hz, %d channel(s), blocksize=%d",
                self.sample_rate,
                self.channels,
                self.block_size,
            )
        except sd.PortAudioError as exc:
            err_msg = str(exc)
            logger.error("Failed to open microphone input stream: %s", exc)
            if "Unanticipated host error" in err_msg or -9999 in getattr(exc, "args", ()):
                raise MicrophonePermissionError(
                    "macOS Microphone access denied or audio device unavailable.\n"
                    "Please check:\n"
                    "  1. System Settings -> Privacy & Security -> Microphone\n"
                    "  2. Ensure your terminal emulator has microphone permissions.\n"
                    f"Details: {exc}"
                ) from exc
            raise MicrophoneError(f"Could not initialize audio input stream: {exc}") from exc
        except Exception as exc:
            raise MicrophoneError(f"Unexpected error starting microphone: {exc}") from exc

    async def stop(self) -> None:
        """Stop microphone capture stream."""
        self._is_running = False
        self._current_rms = 0.0

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.debug("Error while stopping input stream: %s", exc)
            finally:
                self._stream = None
        logger.info("Audio capture stopped")

    async def read_chunk(self) -> bytes:
        """Read the next PCM chunk asynchronously."""
        if not self._is_running:
            raise RuntimeError("AudioCapture is not running")
        return await self._queue.get()

    async def stream_chunks(self) -> AsyncGenerator[bytes, None]:
        """Async generator yielding captured PCM chunks."""
        while self._is_running:
            try:
                chunk = await self.read_chunk()
                yield chunk
            except asyncio.CancelledError:
                break

    async def __aenter__(self) -> "AudioCapture":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
