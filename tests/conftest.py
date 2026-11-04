"""Pytest fixtures for realtime voice assistant tests."""

import pytest
import numpy as np

from src.audio.utils import generate_sine_wave
from src.config import AssistantConfig


@pytest.fixture
def sample_sine_audio_16k() -> bytes:
    """Generate 100ms of 16kHz sine wave PCM16 audio."""
    return generate_sine_wave(frequency_hz=440.0, duration_sec=0.1, sample_rate=16000)


@pytest.fixture
def sample_sine_audio_24k() -> bytes:
    """Generate 100ms of 24kHz sine wave PCM16 audio."""
    return generate_sine_wave(frequency_hz=440.0, duration_sec=0.1, sample_rate=24000)


@pytest.fixture
def silence_pcm_16k() -> bytes:
    """Generate 100ms of silence in 16-bit PCM (16kHz)."""
    return (np.zeros(1600, dtype=np.int16)).tobytes()


@pytest.fixture
def test_config() -> AssistantConfig:
    """Fixture providing a test AssistantConfig."""
    return AssistantConfig(
        api_key="test-api-key",
        model="gemini-3.8-live",
        voice_name="Puck",
        mock=False,
    )


@pytest.fixture
def mock_config() -> AssistantConfig:
    """Fixture providing a mock AssistantConfig."""
    return AssistantConfig(
        mock=True,
        mock_type="echo",
        vad_threshold_db=-36.0,
        barge_in=True,
    )
