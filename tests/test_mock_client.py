"""Unit tests for MockGeminiLiveClient simulation mode and barge-in behavior."""

import asyncio
import pytest

from src.audio.utils import generate_sine_wave
from src.client import (
    AudioReceived,
    Interrupted,
    MockGeminiLiveClient,
    TranscriptReceived,
    TurnComplete,
    VoiceActivity,
)
from src.config import AssistantConfig


@pytest.mark.asyncio
async def test_mock_client_lifecycle():
    """Verify mock client connects and closes cleanly."""
    config = AssistantConfig(mock=True, mock_type="echo")
    client = MockGeminiLiveClient(config)
    assert not client.is_connected

    await client.connect()
    assert client.is_connected

    await client.close()
    assert not client.is_connected


@pytest.mark.asyncio
async def test_mock_client_detects_speech_and_echoes():
    """Verify mock client detects voice activity and generates an echo response."""
    config = AssistantConfig(mock=True, mock_type="echo", vad_threshold_db=-40.0)
    client = MockGeminiLiveClient(config)
    await client.connect()

    # Generate 150ms of audio above -40 dB
    loud_audio = generate_sine_wave(frequency_hz=440.0, duration_sec=0.15, sample_rate=16000, amplitude=0.5)
    await client.send_audio(loud_audio)

    # First event should be VoiceActivity(START)
    event1 = await asyncio.wait_for(client._event_queue.get(), timeout=2.0)
    assert isinstance(event1, VoiceActivity)
    assert event1.event_type == "START"

    event2 = await asyncio.wait_for(client._event_queue.get(), timeout=2.0)
    assert isinstance(event2, TranscriptReceived)
    assert event2.is_user is True

    # Wait for silence detector to trigger turn completion and echo (>0.5s silence)
    events = []
    try:
        while True:
            ev = await asyncio.wait_for(client._event_queue.get(), timeout=1.5)
            events.append(ev)
            if isinstance(ev, TurnComplete):
                break
    except asyncio.TimeoutError:
        pass

    # Should contain AudioReceived chunks at 24kHz
    audio_events = [e for e in events if isinstance(e, AudioReceived)]
    assert len(audio_events) > 0
    assert audio_events[0].sample_rate == 24000

    await client.close()


@pytest.mark.asyncio
async def test_mock_client_barge_in():
    """Verify that speech sent while assistant is playing triggers an immediate Interrupted event."""
    config = AssistantConfig(mock=True, mock_type="sine", vad_threshold_db=-40.0)
    client = MockGeminiLiveClient(config)
    await client.connect()

    # Artificially set assistant playing
    client._mock_player_active = True

    # User speaks while assistant is playing
    loud_audio = generate_sine_wave(frequency_hz=600.0, duration_sec=0.1, sample_rate=16000, amplitude=0.5)
    await client.send_audio(loud_audio)

    # Assistant should be interrupted immediately!
    assert client._mock_player_active is False
    event1 = await asyncio.wait_for(client._event_queue.get(), timeout=1.0)
    assert isinstance(event1, Interrupted)

    await client.close()
