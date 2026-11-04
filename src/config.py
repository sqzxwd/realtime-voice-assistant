"""Configuration settings for realtime-voice-assistant."""

import os
from typing import Literal, Optional
from pydantic import BaseModel, Field


class AssistantConfig(BaseModel):
    """Configuration schema for voice streaming assistant."""

    model: str = Field(
        default="gemini-3.8-live",
        description="Gemini Multimodal Live API model identifier",
    )
    voice_name: str = Field(
        default="Puck",
        description="Prebuilt voice name (e.g. Puck, Charon, Kore, Fenrir, Aoede)",
    )
    system_instruction: str = Field(
        default="You are a helpful, concise real-time voice assistant. Answer directly and naturally.",
        description="System prompt defining assistant persona",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Gemini API Key (can be set via GEMINI_API_KEY environment variable)",
    )
    mock: bool = Field(
        default=False,
        description="Run in mock simulation mode without connecting to Gemini Live API",
    )
    mock_type: Literal["echo", "sine"] = Field(
        default="echo",
        description="Mock response type: 'echo' repeats captured voice, 'sine' generates audio tone",
    )
    input_sample_rate: int = Field(
        default=16000,
        description="Input mic audio sample rate in Hz",
    )
    output_sample_rate: int = Field(
        default=24000,
        description="Output speaker audio sample rate in Hz",
    )
    block_size: int = Field(
        default=1024,
        description="Audio frame block size for capture and player",
    )
    vad_threshold_db: float = Field(
        default=-36.0,
        description="Mic RMS threshold in dB to trigger user speaking / streaming status",
    )
    barge_in: bool = Field(
        default=True,
        description="Enable client & server barge-in (instant audio player clear on speech)",
    )

    @classmethod
    def from_env_and_args(cls, **kwargs) -> "AssistantConfig":
        """Load configuration from environment variables with keyword argument overrides."""
        env_key = os.environ.get("GEMINI_API_KEY")
        if "api_key" not in kwargs or kwargs["api_key"] is None:
            kwargs["api_key"] = env_key
        return cls(**kwargs)
