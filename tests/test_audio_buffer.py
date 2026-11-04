"""Unit tests for AudioPlayer streaming jitter buffer and barge-in clearing."""

import numpy as np
import pytest

from src.audio.player import AudioPlayer
from src.audio.utils import generate_sine_wave


def test_player_initial_state():
    """Verify player initializes in a clean, stopped state."""
    player = AudioPlayer(sample_rate=24000, channels=1, prebuffer_ms=60)
    assert not player.is_running
    assert not player.is_playing
    assert player.buffered_bytes == 0
    assert player.get_current_rms() == 0.0


def test_player_prebuffering_and_playback():
    """Verify audio is held until prebuffer threshold is met, preventing jitter."""
    # At 24kHz mono 16-bit: 1ms = 48 bytes. 60ms = 2880 bytes.
    player = AudioPlayer(sample_rate=24000, channels=1, prebuffer_ms=60)
    
    # Write only 1000 bytes (< 2880 bytes prebuffer)
    chunk_small = generate_sine_wave(frequency_hz=440.0, duration_sec=0.02, sample_rate=24000)
    player.write(chunk_small)
    assert player.buffered_bytes == len(chunk_small)

    outdata = np.zeros((512, 1), dtype=np.int16)
    # Callback should output silence because buffer is still prebuffering
    player._audio_callback(outdata, frames=512, time_info={}, status=None)
    assert np.all(outdata == 0)
    assert player._is_prebuffering is True

    # Now write enough to exceed 2880 bytes
    chunk_large = generate_sine_wave(frequency_hz=440.0, duration_sec=0.1, sample_rate=24000)
    player.write(chunk_large)
    assert player.buffered_bytes > 2880

    # Next callback should now output real audio
    player._audio_callback(outdata, frames=512, time_info={}, status=None)
    assert not np.all(outdata == 0)
    assert player._is_prebuffering is False
    assert player.get_current_rms() > 0.0


def test_barge_in_instant_clear():
    """Crucial requirement: barge-in must instantly flush playback buffer and silence output."""
    player = AudioPlayer(sample_rate=24000, channels=1, prebuffer_ms=20)
    # Fill buffer with 1 second of loud sine audio (48,000 bytes)
    loud_audio = generate_sine_wave(frequency_hz=440.0, duration_sec=1.0, sample_rate=24000, amplitude=0.8)
    player.write(loud_audio)
    player.flush()  # ungate prebuffering

    # Verify audio plays
    outdata = np.zeros((512, 1), dtype=np.int16)
    player._audio_callback(outdata, frames=512, time_info={}, status=None)
    assert not np.all(outdata == 0)
    assert player.buffered_bytes > 0

    # User speaks -> Barge-in event occurs!
    player.clear()

    # Buffer must be immediately 0 bytes
    assert player.buffered_bytes == 0
    assert player.get_current_rms() == 0.0

    # Next callback must output pure silence without error
    outdata.fill(999)  # dirty buffer
    player._audio_callback(outdata, frames=512, time_info={}, status=None)
    assert np.all(outdata == 0)


def test_player_flush():
    """Verify flush ungates prebuffering for trailing audio."""
    player = AudioPlayer(sample_rate=24000, channels=1, prebuffer_ms=100)
    # Small audio chunk that wouldn't normally exceed prebuffer
    small_chunk = generate_sine_wave(frequency_hz=880.0, duration_sec=0.03, sample_rate=24000)
    player.write(small_chunk)
    assert player._is_prebuffering is True

    # Explicit turn end / flush
    player.flush()
    assert player._is_prebuffering is False

    outdata = np.zeros((256, 1), dtype=np.int16)
    player._audio_callback(outdata, frames=256, time_info={}, status=None)
    assert not np.all(outdata == 0)


def test_partial_chunk_underrun_handling():
    """Verify underflow does not crash and zero-pads remaining output buffer."""
    player = AudioPlayer(sample_rate=24000, channels=1, prebuffer_ms=0)
    # Exactly 100 samples (200 bytes)
    partial_audio = (np.ones(100, dtype=np.int16) * 1000).tobytes()
    player.write(partial_audio)
    player.flush()

    outdata = np.zeros((256, 1), dtype=np.int16)
    player._audio_callback(outdata, frames=256, time_info={}, status=None)

    # First 100 frames should be non-zero
    assert np.all(outdata[:100] == 1000)
    # Remaining 156 frames must be 0
    assert np.all(outdata[100:] == 0)
    assert player.buffered_bytes == 0
