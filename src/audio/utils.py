"""Audio utility functions for PCM manipulation, volume measurement, and signal generation."""

import math
from typing import Tuple
import numpy as np


def pcm16_to_numpy(pcm_bytes: bytes, dtype=np.int16) -> np.ndarray:
    """Convert raw 16-bit little-endian PCM bytes into a numpy array."""
    if not pcm_bytes:
        return np.array([], dtype=dtype)
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(dtype)


def numpy_to_pcm16(arr: np.ndarray) -> bytes:
    """Convert a numpy array to raw 16-bit little-endian PCM bytes."""
    if arr.size == 0:
        return b""
    if arr.dtype != np.int16:
        # Clip to int16 range to avoid wrap-around overflow
        clipped = np.clip(arr, -32768, 32767).astype(np.int16)
        return clipped.tobytes()
    return arr.tobytes()


def calculate_rms(pcm_bytes: bytes) -> float:
    """Calculate normalized Root Mean Square (RMS) amplitude in range [0.0, 1.0].
    
    16-bit signed PCM has range [-32768, 32767].
    """
    if not pcm_bytes or len(pcm_bytes) < 2:
        return 0.0
    
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
        
    mean_sq = np.mean(samples ** 2)
    rms_val = np.sqrt(mean_sq)
    # Normalize by max possible int16 amplitude
    return float(min(1.0, rms_val / 32768.0))


def rms_to_db(rms: float, min_db: float = -60.0) -> float:
    """Convert normalized RMS (0.0 to 1.0) to decibels (min_db to 0.0 dB)."""
    if rms <= 1e-5:
        return min_db
    db = 20.0 * math.log10(rms)
    return float(max(min_db, min(0.0, db)))


def render_vu_bar(db: float, width: int = 16, min_db: float = -50.0, max_db: float = 0.0) -> str:
    """Render a terminal-friendly VU meter bar.
    
    Example: [████████░░░░░░░░]
    """
    if db <= min_db:
        ratio = 0.0
    elif db >= max_db:
        ratio = 1.0
    else:
        ratio = (db - min_db) / (max_db - min_db)
    
    filled = int(round(ratio * width))
    empty = width - filled
    return "█" * filled + "░" * empty


def generate_sine_wave(
    frequency_hz: float = 440.0,
    duration_sec: float = 1.0,
    sample_rate: int = 24000,
    amplitude: float = 0.3,
) -> bytes:
    """Generate raw 16-bit PCM sine wave for mock audio responses."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Apply soft attack and release envelope to prevent clicks
    envelope = np.ones_like(t)
    fade_len = min(int(sample_rate * 0.02), len(t) // 2)  # 20ms fade
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        envelope[:fade_len] = fade_in
        envelope[-fade_len:] = fade_out

    waveform = np.sin(2 * np.pi * frequency_hz * t) * amplitude * envelope
    int16_samples = (waveform * 32767.0).astype(np.int16)
    return int16_samples.tobytes()


def resample_pcm16_linear(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Fast linear interpolation resampler for PCM16 audio."""
    if src_rate == dst_rate or not pcm_bytes:
        return pcm_bytes
    
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size == 0:
        return b""
        
    num_dst_samples = int(round(len(samples) * dst_rate / src_rate))
    if num_dst_samples == 0:
        return b""
        
    x_old = np.linspace(0, 1, len(samples))
    x_new = np.linspace(0, 1, num_dst_samples)
    resampled = np.interp(x_new, x_old, samples).astype(np.int16)
    return resampled.tobytes()
