"""Unit tests for AudioCapture and macOS permission handling."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import sounddevice as sd

from src.audio.capture import (
    AudioCapture,
    MicrophoneError,
    MicrophonePermissionError,
)


@pytest.mark.asyncio
async def test_capture_permission_error_handling():
    """Verify that macOS PortAudio permission denials are mapped to MicrophonePermissionError."""
    capture = AudioCapture(sample_rate=16000)

    # Simulate macOS PortAudioError with host error code
    host_error = sd.PortAudioError("Error opening InputStream: Unanticipated host error")
    with patch("sounddevice.InputStream", side_effect=host_error):
        with pytest.raises(MicrophonePermissionError, match="macOS Microphone access denied"):
            await capture.start()

    assert not capture.is_running


@pytest.mark.asyncio
async def test_capture_generic_error_handling():
    """Verify generic PortAudio errors raise MicrophoneError."""
    capture = AudioCapture(sample_rate=16000)

    generic_error = sd.PortAudioError("Invalid sample rate")
    with patch("sounddevice.InputStream", side_effect=generic_error):
        with pytest.raises(MicrophoneError, match="Could not initialize audio input stream"):
            await capture.start()

    assert not capture.is_running


@pytest.mark.asyncio
async def test_capture_callback_queue_and_rms():
    """Verify audio callback pushes PCM into queue and updates RMS atomically."""
    capture = AudioCapture(sample_rate=16000, max_queue_size=5)
    capture._is_running = True
    capture._loop = MagicMock()
    capture._loop.is_closed.return_value = False

    # Create dummy frame: 512 samples
    dummy_frame = np.ones((512, 1), dtype=np.int16) * 1000

    capture._audio_callback(dummy_frame, frames=512, time_info={}, status=sd.CallbackFlags())
    
    assert capture.get_current_rms() > 0.0
    assert capture._loop.call_soon_threadsafe.called


@pytest.mark.asyncio
async def test_capture_queue_overflow_drop():
    """Verify that queue overflow drops oldest chunk to prevent unbounded buffer lag."""
    capture = AudioCapture(sample_rate=16000, max_queue_size=2)
    capture._is_running = True
    
    # Fill queue to capacity
    await capture._queue.put(b"chunk1")
    await capture._queue.put(b"chunk2")
    assert capture._queue.full()

    # Simulate callback running in event loop thread
    class ImmediateLoop:
        def call_soon_threadsafe(self, fn, *args):
            fn(*args)
        def is_closed(self):
            return False

    capture._loop = ImmediateLoop()
    dummy_frame = np.zeros((10, 1), dtype=np.int16)
    capture._audio_callback(dummy_frame, frames=10, time_info={}, status=sd.CallbackFlags())

    # Queue should still have max size 2, but oldest chunk dropped
    assert capture._queue.qsize() == 2
    first_out = await capture.read_chunk()
    assert first_out == b"chunk2"  # chunk1 was dropped
