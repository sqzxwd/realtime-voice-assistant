"""Unit tests for CLI commands and Rich UI state transitions."""

from typer.testing import CliRunner
import pytest

from src.cli import AssistantUIState, app, create_ui_renderable

runner = CliRunner()


def test_cli_help():
    """Verify CLI --help displays program description and flags."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "real-time two-way voice streaming assistant" in result.output
    assert "--mock" in result.output
    assert "--model" in result.output
    assert "--barge-in" in result.output


def test_cli_list_devices():
    """Verify CLI --list-devices displays device table."""
    result = runner.invoke(app, ["--list-devices"])
    assert result.exit_code == 0
    assert "Available Audio Devices" in result.output


def test_cli_missing_api_key_exits_gracefully():
    """Verify running without API key and without --mock warns user and exits with code 1."""
    # Ensure no environment API key leaks in
    result = runner.invoke(app, [], env={"GEMINI_API_KEY": ""})
    assert result.exit_code == 1
    assert "No GEMINI_API_KEY detected" in result.output


def test_cli_invalid_mock_type():
    """Verify invalid --mock-type option is rejected."""
    result = runner.invoke(app, ["--mock", "--mock-type", "invalid_mode"])
    assert result.exit_code == 1
    assert "Invalid --mock-type" in result.output


def test_ui_state_transitions():
    """Verify UI state transitions between LISTENING, STREAMING, SPEAKING, and BARGE-IN."""
    state = AssistantUIState(mode_str="TEST")
    assert state.status == "LISTENING"

    state.status = "STREAMING"
    assert state.status == "STREAMING"

    state.status = "SPEAKING"
    assert state.status == "SPEAKING"

    state.status = "BARGE-IN"
    state.barge_in_count += 1
    assert state.status == "BARGE-IN"
    assert state.barge_in_count == 1

    # Test adding transcripts
    state.add_transcript("User", "Hello")
    state.add_transcript("Assistant", "Hi there")
    assert len(state.transcripts) == 2

    # Render widget
    panel = create_ui_renderable(state)
    assert panel is not None
