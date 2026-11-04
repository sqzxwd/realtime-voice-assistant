"""Audio capture and playback modules."""

from src.audio.capture import AudioCapture, MicrophoneError, MicrophonePermissionError
from src.audio.player import AudioPlayer, PlayerError
from src.audio.utils import (
    calculate_rms,
    generate_sine_wave,
    numpy_to_pcm16,
    pcm16_to_numpy,
    render_vu_bar,
    resample_pcm16_linear,
    rms_to_db,
)

__all__ = [
    "AudioCapture",
    "AudioPlayer",
    "MicrophoneError",
    "MicrophonePermissionError",
    "PlayerError",
    "calculate_rms",
    "generate_sine_wave",
    "numpy_to_pcm16",
    "pcm16_to_numpy",
    "render_vu_bar",
    "resample_pcm16_linear",
    "rms_to_db",
]
