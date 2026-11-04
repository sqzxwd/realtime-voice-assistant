"""Unit tests for PCM audio processing utilities."""

import math
import numpy as np
import pytest

from src.audio.utils import (
    calculate_rms,
    generate_sine_wave,
    numpy_to_pcm16,
    pcm16_to_numpy,
    render_vu_bar,
    resample_pcm16_linear,
    rms_to_db,
)


def test_pcm16_and_numpy_roundtrip():
    """Verify lossless conversion between numpy int16 array and PCM bytes."""
    original = np.array([-32768, -1000, 0, 1000, 32767], dtype=np.int16)
    pcm_bytes = numpy_to_pcm16(original)
    recovered = pcm16_to_numpy(pcm_bytes)

    assert np.array_equal(original, recovered)
    assert len(pcm_bytes) == len(original) * 2


def test_numpy_to_pcm16_clipping():
    """Verify out-of-range floats or int32 are safely clipped to int16 boundaries."""
    overflow = np.array([-50000, 50000], dtype=np.int32)
    pcm_bytes = numpy_to_pcm16(overflow)
    recovered = pcm16_to_numpy(pcm_bytes)

    assert recovered[0] == -32768
    assert recovered[1] == 32767


def test_calculate_rms_silence():
    """Silence should result in zero RMS."""
    silence = np.zeros(1000, dtype=np.int16).tobytes()
    rms = calculate_rms(silence)
    assert rms == 0.0

    empty = b""
    assert calculate_rms(empty) == 0.0


def test_calculate_rms_sine():
    """Sine wave should have predictable RMS ~ amplitude / sqrt(2)."""
    # Amplitude 0.5 sine wave has theoretical RMS ~ 0.5 / sqrt(2) ≈ 0.3535
    sine_pcm = generate_sine_wave(frequency_hz=440.0, duration_sec=0.5, amplitude=0.5)
    rms = calculate_rms(sine_pcm)
    assert 0.30 < rms < 0.40


def test_rms_to_db():
    """Test RMS to dB conversion."""
    assert rms_to_db(0.0, min_db=-60.0) == -60.0
    assert rms_to_db(1.0) == 0.0
    # RMS 0.1 is 20 * log10(0.1) = -20 dB
    assert math.isclose(rms_to_db(0.1), -20.0, abs_tol=0.1)


def test_render_vu_bar():
    """Test VU bar rendering with different dB levels."""
    width = 10
    # Minimum dB -> all empty
    bar_min = render_vu_bar(-60.0, width=width, min_db=-50.0, max_db=0.0)
    assert bar_min == "░" * width

    # Maximum dB -> all filled
    bar_max = render_vu_bar(0.0, width=width, min_db=-50.0, max_db=0.0)
    assert bar_max == "█" * width

    # Halfway (-25 dB)
    bar_mid = render_vu_bar(-25.0, width=width, min_db=-50.0, max_db=0.0)
    assert bar_mid == "█" * 5 + "░" * 5


def test_generate_sine_wave():
    """Test sine wave generation parameters."""
    sample_rate = 24000
    duration = 0.25
    sine = generate_sine_wave(frequency_hz=1000.0, duration_sec=duration, sample_rate=sample_rate)
    
    expected_samples = int(sample_rate * duration)
    assert len(sine) == expected_samples * 2  # 2 bytes per sample (int16)

    arr = pcm16_to_numpy(sine)
    assert arr.max() <= 32767
    assert arr.min() >= -32768


def test_resample_pcm16_linear():
    """Test resampling from 16kHz to 24kHz (1.5x upsampling)."""
    samples_16k = 1600  # 100ms at 16kHz
    sine_16k = generate_sine_wave(frequency_hz=400.0, duration_sec=0.1, sample_rate=16000)
    
    resampled_24k = resample_pcm16_linear(sine_16k, src_rate=16000, dst_rate=24000)
    samples_24k = len(resampled_24k) // 2

    # Should be approximately 2400 samples
    assert abs(samples_24k - 2400) <= 2
