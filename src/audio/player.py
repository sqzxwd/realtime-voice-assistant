"""Asynchronous streaming audio player with jitter buffer and instant barge-in support."""

import logging
import threading
from typing import Optional
import numpy as np
import sounddevice as sd

from src.audio.utils import calculate_rms, rms_to_db

logger = logging.getLogger(__name__)


class PlayerError(Exception):
    """Base exception for audio playback errors."""


class AudioPlayer:
    """Streams raw PCM16 audio to speakers without stuttering, with instant barge-in clear."""

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        block_size: int = 1024,
        prebuffer_ms: int = 60,
        device: Optional[int | str] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device = device
        self.prebuffer_bytes = int((sample_rate * channels * 2) * (prebuffer_ms / 1000.0))

        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._is_prebuffering = True
        self._stream: Optional[sd.OutputStream] = None
        self._is_running = False
        self._current_rms: float = 0.0
        self._active_frames_played: int = 0

    @property
    def is_running(self) -> bool:
        """Return True if the output stream is open and active."""
        return self._is_running

    @property
    def is_playing(self) -> bool:
        """Return True if there is audio actively buffered or playing."""
        with self._lock:
            return len(self._buffer) > 0 or self._current_rms > 0.01

    @property
    def buffered_bytes(self) -> int:
        """Return current number of bytes queued in the audio buffer."""
        with self._lock:
            return len(self._buffer)

    def get_current_rms(self) -> float:
        """Get the latest normalized RMS output level [0.0, 1.0]."""
        return self._current_rms

    def get_current_db(self, min_db: float = -60.0) -> float:
        """Get the latest audio output level in decibels [min_db, 0.0]."""
        return rms_to_db(self._current_rms, min_db=min_db)

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """PortAudio C callback for writing audio to output device."""
        if status:
            logger.debug("Audio player status flag: %s", status)

        bytes_needed = frames * self.channels * 2

        with self._lock:
            # Check pre-buffering condition to eliminate jitter on start
            if self._is_prebuffering:
                if len(self._buffer) < self.prebuffer_bytes:
                    outdata.fill(0)
                    self._current_rms = 0.0
                    return
                self._is_prebuffering = False

            avail_bytes = len(self._buffer)

            if avail_bytes >= bytes_needed:
                # Full chunk available
                chunk = bytes(self._buffer[:bytes_needed])
                del self._buffer[:bytes_needed]
                audio_np = np.frombuffer(chunk, dtype=np.int16).reshape(-1, self.channels)
                outdata[:] = audio_np
                self._current_rms = calculate_rms(chunk)
                self._active_frames_played += frames
            elif avail_bytes > 0:
                # Partial chunk available (end of audio burst)
                chunk = bytes(self._buffer)
                self._buffer.clear()
                self._is_prebuffering = True  # reset prebuffering for next sentence
                partial_np = np.frombuffer(chunk, dtype=np.int16).reshape(-1, self.channels)
                frames_avail = len(partial_np)
                outdata[:frames_avail] = partial_np
                outdata[frames_avail:].fill(0)
                self._current_rms = calculate_rms(chunk)
                self._active_frames_played += frames_avail
            else:
                # Buffer empty
                outdata.fill(0)
                self._current_rms = 0.0
                self._is_prebuffering = True

    def write(self, pcm_data: bytes) -> None:
        """Enqueue PCM16 audio bytes into the playback ring buffer."""
        if not pcm_data:
            return
        with self._lock:
            self._buffer.extend(pcm_data)

    def clear(self) -> None:
        """Instant barge-in clear: immediately flushes playback buffer and silences output."""
        with self._lock:
            self._buffer.clear()
            self._is_prebuffering = True
            self._current_rms = 0.0
        logger.debug("Playback buffer flushed (barge-in)")

    def flush(self) -> None:
        """Ungate prebuffering so any remaining data in the buffer plays out immediately."""
        with self._lock:
            self._is_prebuffering = False

    def start(self) -> None:
        """Start the PortAudio output stream."""
        if self._is_running:
            return

        with self._lock:
            self._buffer.clear()
            self._is_prebuffering = True
            self._current_rms = 0.0

        try:
            self._stream = sd.OutputStream(
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
                "Audio player started: %d Hz, %d channel(s), blocksize=%d",
                self.sample_rate,
                self.channels,
                self.block_size,
            )
        except Exception as exc:
            raise PlayerError(f"Failed to start audio output stream: {exc}") from exc

    def stop(self) -> None:
        """Stop and close the audio output stream."""
        self._is_running = False
        self.clear()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.debug("Error stopping output stream: %s", exc)
            finally:
                self._stream = None
        logger.info("Audio player stopped")

    def __enter__(self) -> "AudioPlayer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
