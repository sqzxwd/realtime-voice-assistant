"""Unit tests for Gemini Multimodal Live API WebSocket protocol handling."""

import base64
import json
import pytest

from src.client import (
    AudioReceived,
    GeminiLiveClient,
    GeminiLiveProtocolError,
    Interrupted,
    TranscriptReceived,
    TurnComplete,
    VoiceActivity,
)
from src.config import AssistantConfig


def test_ws_url_generation(test_config: AssistantConfig):
    """Verify authenticated WebSocket URL format."""
    client = GeminiLiveClient(test_config)
    url = client._get_ws_url()
    assert "wss://generativelanguage.googleapis.com" in url
    assert "key=test-api-key" in url


def test_missing_api_key_raises_error():
    """Verify missing API key raises GeminiLiveProtocolError."""
    config = AssistantConfig(api_key=None, mock=False)
    client = GeminiLiveClient(config)
    with pytest.raises(GeminiLiveProtocolError, match="Gemini API Key is missing"):
        client._get_ws_url()


@pytest.mark.asyncio
async def test_parse_audio_response(test_config: AssistantConfig):
    """Verify parsing of serverContent modelTurn audio chunks."""
    client = GeminiLiveClient(test_config)
    
    raw_pcm = b"\x01\x02\x03\x04" * 100
    b64_data = base64.b64encode(raw_pcm).decode("ascii")

    server_msg = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "audio/pcm;rate=24000",
                            "data": b64_data,
                        }
                    }
                ]
            }
        }
    }

    await client._parse_and_dispatch(server_msg)
    event = await client._event_queue.get()

    assert isinstance(event, AudioReceived)
    assert event.pcm_data == raw_pcm
    assert event.sample_rate == 24000


@pytest.mark.asyncio
async def test_parse_interrupted_barge_in(test_config: AssistantConfig):
    """Verify parsing of serverContent.interrupted barge-in signal."""
    client = GeminiLiveClient(test_config)
    server_msg = {
        "serverContent": {
            "interrupted": True,
        }
    }

    await client._parse_and_dispatch(server_msg)
    event = await client._event_queue.get()
    assert isinstance(event, Interrupted)


@pytest.mark.asyncio
async def test_parse_transcriptions(test_config: AssistantConfig):
    """Verify parsing of user and assistant transcription messages."""
    client = GeminiLiveClient(test_config)
    server_msg = {
        "serverContent": {
            "inputTranscription": {"text": "Hello assistant"},
            "outputTranscription": {"text": "Hello, how can I help you?"},
        }
    }

    await client._parse_and_dispatch(server_msg)
    event1 = await client._event_queue.get()
    assert isinstance(event1, TranscriptReceived)
    assert event1.text == "Hello assistant"
    assert event1.is_user is True

    event2 = await client._event_queue.get()
    assert isinstance(event2, TranscriptReceived)
    assert event2.text == "Hello, how can I help you?"
    assert event2.is_user is False


@pytest.mark.asyncio
async def test_parse_voice_activity(test_config: AssistantConfig):
    """Verify parsing of server voiceActivity signals."""
    client = GeminiLiveClient(test_config)
    
    msg_start = {"voiceActivity": {"type": "VOICE_ACTIVITY_START"}}
    await client._parse_and_dispatch(msg_start)
    e1 = await client._event_queue.get()
    assert isinstance(e1, VoiceActivity)
    assert e1.event_type == "START"

    msg_end = {"voiceActivity": {"type": "VOICE_ACTIVITY_END"}}
    await client._parse_and_dispatch(msg_end)
    e2 = await client._event_queue.get()
    assert isinstance(e2, VoiceActivity)
    assert e2.event_type == "END"


@pytest.mark.asyncio
async def test_parse_turn_complete(test_config: AssistantConfig):
    """Verify parsing of turnComplete signal."""
    client = GeminiLiveClient(test_config)
    server_msg = {
        "serverContent": {
            "turnComplete": True,
        }
    }

    await client._parse_and_dispatch(server_msg)
    event = await client._event_queue.get()
    assert isinstance(event, TurnComplete)
